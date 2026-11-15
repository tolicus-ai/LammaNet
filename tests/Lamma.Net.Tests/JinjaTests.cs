using LammaNet.Templates.Jinja;

namespace LammaNet.Tests;

public class JinjaTests
{
    // expected outputs were rendered by Python jinja2 with strftime_now pinned to this date
    private static readonly DateTime ReferenceDate = new(2026, 10, 12);

    private static string DataDir => Path.Combine(AppContext.BaseDirectory, "TestData", "templates");

    public static IEnumerable<object[]> Cases() =>
        Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "TestData", "templates"), "*.expected")
            .Select(f => new object[] { Path.GetFileName(f) });

    /// <summary>Renders real model chat templates and compares with Python jinja2 (HF settings) output.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesPythonJinja(string expectedFile)
    {
        // <template>.jinja.<ctx>.expected
        var name = Path.GetFileNameWithoutExtension(expectedFile);
        int dot = name.LastIndexOf('.');
        string templateFile = name[..dot], ctx = name[(dot + 1)..];
        var tpl = new JinjaTemplate(File.ReadAllText(Path.Combine(DataDir, templateFile)));
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, $"ctx_{ctx}.json")));
        var vars = doc.RootElement.EnumerateObject().ToDictionary(p => p.Name, p => JinjaValue.FromJson(p.Value.Clone()));
        string expected = File.ReadAllText(Path.Combine(DataDir, expectedFile));
        if (expected == "ERROR")
        {
            Assert.Throws<JinjaException>(() => tpl.Render(vars, ReferenceDate));
            return;
        }
        Assert.Equal(expected, tpl.Render(vars, ReferenceDate));
    }

    [Theory]
    [InlineData("{{ 'a' ~ 1 ~ none }}", "a1None")]
    [InlineData("{% for x in [1,2,3] if x > 1 %}{{ x }}{% if not loop.last %},{% endif %}{% endfor %}", "2,3")]
    [InlineData("{{ [3,1,2] | sort | join('-') }}", "1-2-3")]
    [InlineData("{{ {'b': 1, 'a': 2} | tojson }}", "{\"b\": 1, \"a\": 2}")]
    [InlineData("{{ 'Hello World'[::-1] }}", "dlroW olleH")]
    [InlineData("{% set ns = namespace(x=0) %}{% for i in range(5) %}{% set ns.x = ns.x + i %}{% endfor %}{{ ns.x }}", "10")]
    [InlineData("{% macro f(a, b='y') %}{{ a }}{{ b }}{% endmacro %}{{ f('x') }}{{ f('x', b='z') }}", "xyxz")]
    [InlineData("{{ 7 // 2 }} {{ 7 % 3 }} {{ 2 ** 3 }} {{ 1.5 + 1 }}", "3 1 8 2.5")]
    [InlineData("{{ 'a,b,,c'.split(',') | length }}", "4")]
    [InlineData("{% for k, v in {'x': 1, 'y': 2} | dictsort %}{{ k }}={{ v }};{% endfor %}", "x=1;y=2;")]
    [InlineData("{{ x is defined }} {{ y | default('d') }}", "False d")]
    [InlineData("{%- for i in range(3) -%}{%- if i == 1 -%}{%- continue -%}{%- endif -%}{{ i }}{%- endfor -%}", "02")]
    public void Expressions(string template, string expected)
    {
        Assert.Equal(expected, new JinjaTemplate(template).Render(new Dictionary<string, object?>()));
    }
}
