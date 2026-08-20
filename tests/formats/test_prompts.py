def test_the_alternatives_instruction_leads_rather_than_trails():
    """Measured on five real parents, five samples per wording: as an option
    last in a list of rules the marker was used 0/5 times and a reply carried
    one candidate; stated up front as the shape of the reply it was used 5/5 and
    a reply carried 2.6 (Fisher exact p = 0.008). The wording was the whole
    difference, so the instruction has to come first."""
    from vectrify.formats.prompts import diff_format_instructions

    text = diff_format_instructions("SVG", unit="fragment", subject="SVG")
    head = text.split("Rules:")[0]
    assert "===ALTERNATIVE===" in head
