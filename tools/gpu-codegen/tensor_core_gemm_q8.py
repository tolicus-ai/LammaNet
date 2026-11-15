import os
import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
# Generates the int8 tensor-core GEMMs (TcGemmQ4K, TcGemmQ40, TcGemmQ80). Usage: python tensor_core_gemm_q8.py [out.cs]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'src', 'Lamma.Net.Gpu', 'TensorCoreGemmQ8.cs')

BM, BN, BK = 64, 128, 64       # block tile: weight rows x tokens x columns per k-step (two 32-weight blocks)
WM, WN = 2, 4                  # warp grid
THREADS = 32 * WM * WN
WTM, WTN = BM // WM, BN // WN  # 32 x 32 per warp
MT, NT = WTM // 16, WTN // 8   # 2 x 4 mma tiles
LDB = 20                       # words per shared token row (16 used): conflict-free fragment loads
NACC = 4 * MT * NT
# sub-blocks computed before the prefetched step is stored to the free buffer (0: shortest register lifetime,
# 128 registers -> 2 blocks per SM)
STORE_AFTER = int(os.environ.get('Q8_STORE_AFTER', '0'))

# Per format: weight words per row and k-step (A_WORDS), padded shared row (LDA, conflict-free for rows g..g+7),
# threads that load the per-row scales (SCALE_TID0..+64), mma operand types, whether a min term exists.
FORMATS = {
    'Q4K': dict(A_WORDS=8, LDA=12, SCALE_TID0=128, ATYPE='u8', MIN=True,
                doc='Q4_K: the 32 nibble bytes of a 64-weight chunk (low nibbles: sub-block 2c, high: 2c+1); '
                    'y += d·sc·dx·Σq·x − dmin·m·dx·Σx'),
    'Q40': dict(A_WORDS=8, LDA=12, SCALE_TID0=128, ATYPE='u8', MIN=True,
                doc='Q4_0: two 16-byte blocks (low nibbles: weights 0-15, high: 16-31); y += d·dx·Σq·x − 8·d·dx·Σx'),
    'Q80': dict(A_WORDS=16, LDA=20, SCALE_TID0=0, ATYPE='s8', MIN=False,
                doc='Q8_0: two 32-byte blocks of signed bytes; y += d·dx·Σq·x'),
}

L = []
A = L.append


def indent(start, by):
    for i in range(start, len(L)):
        L[i] = by + L[i]


def kernel(name, fmt):
    lda, aw4, stid = fmt['LDA'], fmt['A_WORDS'] // 4, fmt['SCALE_TID0']
    wl = BM * aw4   # 16-byte weight loads per step
    A('    /// <summary>')
    A(f'    /// int8 tensor-core GEMM for prompt processing (sm_80+), as llama.cpp\'s MMQ. {fmt["doc"]}.')
    A('    /// Activations are q8 blocks (<see cref="QuantizeQ8"/>); mma.sync m16n8k32 per 32-weight block into int32, then')
    A('    /// the block scales (and mins) in fp32, once per block instead of per weight.')
    A(f'    /// A {THREADS}-thread block computes {BM} rows x {BN} tokens, warp w {WTM} x {WTN} ({MT}x{NT} tiles); each k-step')
    A('    /// covers 64 weights (raw weight bytes, per-row scales, int8 activations and their block scales in double-buffered')
    A('    /// shared memory; the next step is loaded and stored to the free buffer before the current mma).')
    A('    /// </summary>')
    A(f'    public static void TcGemm{name}(ArrayView<uint> w, ArrayView<float> f, ArrayView<uint> xq, ArrayView<Float2> xd, ArrayView<float> y, GemmParams p)')
    A('    {')
    A(f'        var sa = SharedMemory.Allocate<uint>(2 * {BM} * {lda});')
    A(f'        var ss = SharedMemory.Allocate<float>(2 * {BM} * 4);')
    A(f'        var sb = SharedMemory.Allocate<uint>(2 * {BN} * {LDB});')
    A(f'        var sx = SharedMemory.Allocate<float>(2 * {BN} * 4);')
    A('        var sa4 = sa.Cast<UInt4>();')
    A('        var ss4 = ss.Cast<Float4>();')
    A('        var sb4 = sb.Cast<UInt4>();')
    A('        var sx4 = sx.Cast<Float4>();')
    A('        var w4 = w.Cast<UInt4>();')
    A('        var xq4 = xq.Cast<UInt4>();')
    A('        var xd4 = xd.Cast<Float4>();')
    A('        int tid = Group.IdxX, lane = tid & 31, warp = tid >> 5;')
    A(f'        int tilesM = (p.Rows + {BM} - 1) / {BM};')
    A(f'        int row0 = (Grid.IdxX % tilesM) * {BM}, tok0 = (Grid.IdxX / tilesM) * {BN};')
    A(f'        int wr = (warp / {WN}) * {WTM}, wc = (warp % {WN}) * {WTN};')
    A('        int g = lane >> 2, t4 = lane & 3;')
    A('        int xqs4 = p.Cols / 16, xds4 = p.Cols / 64;   // 16-byte groups per token row of xq / xd')
    zero = ''.join(f'\\n\\tmov.f32 td{i}, 0f00000000;' for i in range(NACC))
    A(f'        CudaAsm.Emit(".reg .f32 td<{NACC}>, tf<{NACC}>;\\n\\t.reg .s32 ti<{NACC}>;{zero}");')
    A(f'        // load slots: weights (tid < {wl}): row tid / {aw4}, 16-byte part tid % {aw4}; scales (tid {stid}..{stid + BM - 1}): row tid - {stid};')
    A('        // activations: every thread 2 x 16 bytes; token scales: tid < 128')
    A(f'        int ar = tid / {aw4}, ah = tid % {aw4}, hr = tid - {stid};')
    A(f'        bool aOk = tid < {wl} && row0 + ar < p.Rows, hOk = hr >= 0 && hr < {BM} && row0 + hr < p.Rows;')
    A('        long aBase = aOk ? (p.WWord + (long)(row0 + ar) * p.RowWords) >> 2 : 0;')
    A('        long hWord = hOk ? p.WWord + (long)(row0 + hr) * p.RowWords : 0;')
    A('        int bt0 = tid >> 2, bq0 = tid & 3, bt1 = (tid + 256) >> 2, bq1 = (tid + 256) & 3;')
    A('        bool b0Ok = tok0 + bt0 < p.NTok, b1Ok = tok0 + bt1 < p.NTok, xOk = tid < 128 && tok0 + tid < p.NTok;')
    A('        long b0Base = (long)(tok0 + bt0) * xqs4 + bq0, b1Base = (long)(tok0 + bt1) * xqs4 + bq1, xBase = (long)(tok0 + tid) * xds4;')

    def loads(k, sfx):
        A(f'        UInt4 la{sfx} = default; Float4 ls{sfx} = default;')
        if name == 'Q4K':
            A(f'        if (aOk) la{sfx} = w4[aBase + (({k}) >> 8) * 9L + 1 + 2 * ((({k}) >> 6) & 3) + ah];')
            A('        if (hOk)')
            A('        {')
            A(f'            int c = (({k}) >> 6) & 3;')
            A(f'            UInt4 h = w4[(hWord >> 2) + (({k}) >> 8) * 9L];')
            A('            float dd = HalfBits((int)(h.X & 0xFFFF)), dmin = HalfBits((int)(h.X >> 16));')
            A('            ScaleMinK4(2 * c, h.Y, h.Z, h.W, out int sc0, out int m0);')
            A('            ScaleMinK4(2 * c + 1, h.Y, h.Z, h.W, out int sc1, out int m1);')
            A(f'            ls{sfx} = new Float4 {{ X = dd * sc0, Y = dd * sc1, Z = -dmin * m0, W = -dmin * m1 }};')
            A('        }')
        elif name == 'Q40':
            A(f'        if (aOk) la{sfx} = w4[aBase + ({k}) / 32 + ah];')
            A('        if (hOk)')
            A('        {')
            A(f'            uint sw = w[hWord + (long)(p.Cols >> 5) * 4 + ({k}) / 64];')
            A('            float d0 = HalfBits((int)(sw & 0xFFFF)), d1 = HalfBits((int)(sw >> 16));')
            A(f'            ls{sfx} = new Float4 {{ X = d0, Y = d1, Z = -8f * d0, W = -8f * d1 }};')
            A('        }')
        else:
            A(f'        if (aOk) la{sfx} = w4[aBase + ({k}) / 16 + ah];')
            A('        if (hOk)')
            A('        {')
            A(f'            uint sw = w[hWord + (long)(p.Cols >> 5) * 8 + ({k}) / 64];')
            A(f'            ls{sfx} = new Float4 {{ X = HalfBits((int)(sw & 0xFFFF)), Y = HalfBits((int)(sw >> 16)) }};')
            A('        }')
        A(f'        UInt4 lb0{sfx} = b0Ok ? xq4[b0Base + ({k}) / 16] : default;')
        A(f'        UInt4 lb1{sfx} = b1Ok ? xq4[b1Base + ({k}) / 16] : default;')
        A(f'        Float4 lx{sfx} = xOk ? xd4[xBase + ({k}) / 64] : default;')

    def stores(buf, sfx):
        A(f'        if (tid < {wl}) sa4[(({buf}) * {BM} + ar) * {lda // 4} + ah] = la{sfx};')
        A(f'        if (hr >= 0 && hr < {BM}) ss4[({buf}) * {BM} + hr] = ls{sfx};')
        A(f'        sb4[(({buf}) * {BN} + bt0) * {LDB // 4} + bq0] = lb0{sfx};')
        A(f'        sb4[(({buf}) * {BN} + bt1) * {LDB // 4} + bq1] = lb1{sfx};')
        A(f'        if (tid < 128) sx4[({buf}) * {BN} + tid] = lx{sfx};')

    def store_next():
        # the other buffer is free (every warp passed the previous barrier): the prefetch need not stay in registers
        A('            if (more)')
        A('            {')
        start = len(L)
        stores('buf ^ 1', 'n')
        indent(start, '        ')
        A('            }')

    A('        // prologue: step 0 into buffer 0')
    loads('0', '0')
    stores('0', '0')
    A('        Group.Barrier();')
    A('        for (int k0 = 0, buf = 0; k0 < p.Cols; k0 += 64, buf ^= 1)')
    A('        {')
    A('            bool more = k0 + 64 < p.Cols;')
    A('            int kn = more ? k0 + 64 : k0;')
    start = len(L)
    loads('kn', 'n')
    indent(start, '    ')
    A(f'            int abase = buf * {BM} * {lda}, sbase = buf * {BM} * 4, bbase = buf * {BN} * {LDB}, xbase = buf * {BN} * 4;')
    for mt in range(MT):
        A(f'            int ra{mt} = abase + (wr + {16 * mt} + g) * {lda} + t4;')
    if name == 'Q4K':
        # both sub-blocks use the same bytes (low / high nibbles)
        for mt in range(MT):
            A(f'            uint aw{mt}_0 = sa[ra{mt}], aw{mt}_1 = sa[ra{mt} + 8 * {lda}], aw{mt}_2 = sa[ra{mt} + 4], aw{mt}_3 = sa[ra{mt} + 8 * {lda} + 4];')
    for s in range(2):
        if s == STORE_AFTER:
            store_next()
        A(f'            // block {s} of the step: activation words {8 * s}..{8 * s + 7}')
        A('            {')
        for mt in range(MT):
            if name == 'Q4K':
                sh = ' >> 4' if s else ''
                A(f'            ulong fa{mt}_01 = ((aw{mt}_0{sh}) & 0x0F0F0F0Fu) | ((ulong)((aw{mt}_1{sh}) & 0x0F0F0F0Fu) << 32);')
                A(f'            ulong fa{mt}_23 = ((aw{mt}_2{sh}) & 0x0F0F0F0Fu) | ((ulong)((aw{mt}_3{sh}) & 0x0F0F0F0Fu) << 32);')
            elif name == 'Q40':
                # weights k and k + 16 are the low / high nibbles of the same byte
                A(f'            uint bw{mt}_0 = sa[ra{mt} + {4 * s}], bw{mt}_1 = sa[ra{mt} + 8 * {lda} + {4 * s}];')
                A(f'            ulong fa{mt}_01 = (bw{mt}_0 & 0x0F0F0F0Fu) | ((ulong)(bw{mt}_1 & 0x0F0F0F0Fu) << 32);')
                A(f'            ulong fa{mt}_23 = ((bw{mt}_0 >> 4) & 0x0F0F0F0Fu) | ((ulong)((bw{mt}_1 >> 4) & 0x0F0F0F0Fu) << 32);')
            else:
                A(f'            ulong fa{mt}_01 = sa[ra{mt} + {8 * s}] | ((ulong)sa[ra{mt} + 8 * {lda} + {8 * s}] << 32);')
                A(f'            ulong fa{mt}_23 = sa[ra{mt} + {8 * s + 4}] | ((ulong)sa[ra{mt} + 8 * {lda} + {8 * s + 4}] << 32);')
            A(f'            int rs{mt} = sbase + (wr + {16 * mt} + g) * 4;')
            A(f'            ulong rd{mt} = Pack2(ss[rs{mt} + {s}], ss[rs{mt} + 32 + {s}]);   // block scale of rows g, g + 8')
            if fmt['MIN']:
                A(f'            ulong rm{mt} = Pack2(ss[rs{mt} + {2 + s}], ss[rs{mt} + 32 + {2 + s}]);   // −min')
        for nt in range(NT):
            A(f'            int rb{nt} = bbase + (wc + {8 * nt} + g) * {LDB} + {8 * s} + t4;')
            A(f'            ulong fb{nt} = sb[rb{nt}] | ((ulong)sb[rb{nt} + 4] << 32);')
            A(f'            int cx{nt} = xbase + (wc + {8 * nt} + 2 * t4) * 4 + {2 * s};')
            A(f'            ulong cd{nt} = Pack2(sx[cx{nt}], sx[cx{nt} + 4]);       // dx of tokens 2·t4, 2·t4 + 1')
            if fmt['MIN']:
                A(f'            ulong cs{nt} = Pack2(sx[cx{nt} + 1], sx[cx{nt} + 5]);   // dx·Σq')
        for mt in range(MT):
            for nt in range(NT):
                t = mt * NT + nt
                r = [4 * t + i for i in range(4)]
                regs = 'qa<4>, qb<2>, qrd<2>, qcd<2>, qz' + (', qrm<2>, qcs<2>' if fmt['MIN'] else '')
                asm = (f'{{\\n\\t.reg .b32 {regs};\\n\\tmov.b32 qz, 0;\\n\\t'
                       'mov.b64 {qa0, qa1}, %0;\\n\\tmov.b64 {qa2, qa3}, %1;\\n\\tmov.b64 {qb0, qb1}, %2;\\n\\t'
                       'mov.b64 {qrd0, qrd1}, %3;\\n\\tmov.b64 {qcd0, qcd1}, %4;\\n\\t')
                if fmt['MIN']:
                    asm += 'mov.b64 {qrm0, qrm1}, %5;\\n\\tmov.b64 {qcs0, qcs1}, %6;\\n\\t'
                asm += (f'mma.sync.aligned.m16n8k32.row.col.s32.{fmt["ATYPE"]}.s8.s32 {{ti{r[0]},ti{r[1]},ti{r[2]},ti{r[3]}}}, '
                        '{qa0,qa1,qa2,qa3}, {qb0,qb1}, {qz,qz,qz,qz};\\n\\t')
                # element i (C fragment): row half i >> 1 (g, g + 8), token i & 1
                for i in range(4):
                    rh, cc = i >> 1, i & 1
                    asm += (f'cvt.rn.f32.s32 tf{r[i]}, ti{r[i]};\\n\\t'
                            f'mul.f32 tf{r[i]}, tf{r[i]}, qrd{rh};\\n\\t'
                            f'fma.rn.f32 td{r[i]}, tf{r[i]}, qcd{cc}, td{r[i]};\\n\\t')
                    if fmt['MIN']:
                        asm += f'fma.rn.f32 td{r[i]}, qrm{rh}, qcs{cc}, td{r[i]};\\n\\t'
                asm += '}'
                args = f'fa{mt}_01, fa{mt}_23, fb{nt}, rd{mt}, cd{nt}' + (f', rm{mt}, cs{nt}' if fmt['MIN'] else '')
                A(f'            CudaAsm.Emit("{asm}", {args});')
        A('            }')
    if STORE_AFTER >= 2:
        store_next()
    A('            Group.Barrier();')
    A('        }')
    for mt in range(MT):
        for nt in range(NT):
            t = mt * NT + nt
            A('        {')
            for i in range(4):
                A(f'            CudaAsm.Emit("mov.f32 %0, td{4 * t + i};", out float c{i});')
            A(f'            int r = row0 + wr + {16 * mt} + g, t = tok0 + wc + {8 * nt} + 2 * t4;')
            A('            TcStore(f, y, p, r, t, c0, c1);')
            A('            TcStore(f, y, p, r + 8, t, c2, c3);')
            A('        }')
    A('    }')
    A('')


A('using ILGPU;')
A('using ILGPU.Runtime;')
A('using ILGPU.Runtime.Cuda;')
A('')
A('namespace LammaNet.Gpu;')
A('')
A('// <auto-generated> by tools/gpu-codegen/tensor_core_gemm_q8.py: the PTX statements name fixed accumulator registers,')
A(f'// so each of the warp\'s {MT * NT} (16x8) tiles has its own literal statements.')
A('')
A('internal static partial class Kernels')
A('{')
A(f'    public const int TcQ8BM = {BM}, TcQ8BN = {BN}, TcQ8Threads = {THREADS};')
A('')
for name, fmt in FORMATS.items():
    kernel(name, fmt)
A('    /// <summary>Two floats as one 64-bit operand (lo, hi) for inline PTX (mov.b64 {lo, hi}).</summary>')
A('    private static ulong Pack2(float lo, float hi) => (uint)Interop.FloatAsInt(lo) | ((ulong)(uint)Interop.FloatAsInt(hi) << 32);')
A('}')

with open(OUT, 'w', newline='\r\n', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')
print('wrote', OUT)
