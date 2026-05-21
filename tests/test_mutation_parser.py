from ciis_binderbench.features import aa_delta_features, parse_mutation_token


def test_parse_mutation_token():
    parsed = parse_mutation_token("RA45G")
    assert parsed["wt"] == "R"
    assert parsed["chain"] == "A"
    assert parsed["resnum"] == 45
    assert parsed["mut"] == "G"


def test_delta_features_nonzero():
    delta = aa_delta_features("R", "G")
    assert delta["d_mass"] < 0
    assert delta["d_charge"] < 0
