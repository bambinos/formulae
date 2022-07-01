from formulae.transforms import Surv

def test_Surv():
    assert Surv([10], [0]).eval().shape[0] == 1
    assert Surv([10], [0]).eval().shape[1] == 2
    assert Surv([10, 20, 30], [0, 1, 1]).eval().shape[0] == 3
    assert Surv([10, 20, 30], [0, 1, 1]).eval().shape[1] == 2

    assert Surv([10], [0]).eval()[0][0] == 10
    assert Surv([10], [0]).eval()[0][1] == 0
