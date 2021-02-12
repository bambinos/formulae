# see https://github.com/pydata/patsy/blob/master/patsy/redundancy.py
def _sorted_subsets(tupl):
    def helper(seq):
        if not seq:
            yield ()
        else:
            obj = seq[0]
            for subset in _sorted_subsets(seq[1:]):
                yield subset
                yield (obj,) + subset

    expanded = list(enumerate(tupl))
    expanded_subsets = list(helper(expanded))
    expanded_subsets.sort()
    expanded_subsets.sort(key=len)

    for subset in expanded_subsets:
        yield tuple([obj for (idx, obj) in subset])


class ExpandedFactor:
    """An factor with an additional annotation for whether it is coded
    full-rank (includes_intercept=True) or not."""

    def __init__(self, includes_intercept, factor):
        self.includes_intercept = includes_intercept
        self.factor = factor

    def __hash__(self):
        return hash((ExpandedFactor, self.includes_intercept, self.factor))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return (
                self.includes_intercept == other.includes_intercept and self.factor == other.factor
            )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.includes_intercept:
            suffix = "+"
        else:
            suffix = "-"
        return "%r%s" % (self.factor, suffix)


# And a collection of Terms make up an EvalFactor
class Subterm:
    """Representation of a Subterm contained within a Term.

    The collection of Subterms of a Term clearly represents all the
    vector spaces in consideration.

    Parameters
    ----------
    efactors: set-like
        A set of one or more ExpandedFactor involved in the Subterm
    """

    def __init__(self, efactors):
        self.efactors = frozenset(efactors)

    def can_absorb(self, other):
        is_one_element_smaller = len(self.efactors) - len(other.efactors) == 1
        is_contained_within_self = self.efactors.issuperset(other.efactors)
        return is_one_element_smaller and is_contained_within_self

    def absorb(self, other):
        """Create new Subterm from the simplification (absorption) of self with another Subterm

        Basically, it returns a new Subterm where 'efactors' is composed of
        * other.efactors
          AND
        * the ExpandedFactor resulting from the  difference between
          self.efactors and other.efactors with the 'includes_intercept' set to True


        Examples
        ----------
        * [a-, b-] "absorb" [a-] -> [a-, b+]
        * [a-, b-] "absorb" [b-] -> [a+, b-]
        * [a-, b-, c-] "absorb" [a-, b-] -> [a-, b-, c+]

        """

        diff = self.efactors.difference(other.efactors)
        assert len(diff) == 1
        efactor = list(diff)[0]
        assert not efactor.includes_intercept
        new_factors = set(other.efactors)
        new_factors.add(ExpandedFactor(True, efactor.factor))
        return Subterm(new_factors)

    def __hash__(self):
        return hash((Subterm, self.efactors))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.efactors == other.efactors

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, list(self.efactors))


class ExpandedTerm:
    def __init__(self, name, components):
        self.name = name
        self.components = components

    def pick_contrast(self, used_subterms):
        """Obtain constrasts for a given term

        Parameters
        ----------
        used_subterms: set
            A set of Subterms that have already been used so they are discarded here.
            This object is modified in-place!
        """

        self.subterms = []
        components = self.components

        for subset in _sorted_subsets(components):
            subterm = Subterm([ExpandedFactor(False, f) for f in subset])
            if subterm not in used_subterms:
                self.subterms.append(subterm)
        # used_subterms is modified in-place
        used_subterms.update(self.subterms)
        self.simplify_subterms()
        factor_codings = []
        for subterm in self.subterms:
            factor_coding = {}
            for expanded in subterm.efactors:
                factor_coding[expanded.factor] = expanded.includes_intercept
            factor_codings.append(factor_coding)
        return factor_codings

    def _simplify_subterm(self):
        # modifies self.subterms
        for short_i, short_subterm in enumerate(self.subterms):
            for long_i, long_subterm in enumerate(self.subterms[short_i + 1 :]):
                if long_subterm.can_absorb(short_subterm):
                    new_subterm = long_subterm.absorb(short_subterm)
                    self.subterms[short_i + 1 + long_i] = new_subterm
                    self.subterms.pop(short_i)
                    return True
        return False

    def simplify_subterms(self):
        while self._simplify_subterm():
            pass
        return self.subterms


def pick_contrasts(group):
    """Determines whether each term is encoded with "n" or "n-1" dummies

    Parameters
    ----------
    terms: ModelTerms
        A set of one or more ExpandedFactor involved in the Subterm
    """

    used_subterms = set()
    codings = dict()

    for name, components in group.items():
        codings[name] = ExpandedTerm(name, components).pick_contrast(used_subterms)

    return codings
