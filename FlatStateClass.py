''' FlatStateClass.py: Contains the FlatState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class FlatState(State):
    ''' Class for Flat States '''

    def __init__(self, xr, yr, u, r, d, l, xg, yg):
        State.__init__(self, data=[xr, yr, u, r, d, l, xg, yg])
        self.xr = round(xr, 5)
        self.yr = round(yr, 5)
        self.u = u
        self.r = r
        self.d = d
        self.l = l
        self.xg = round(xg, 5)
        self.yg = round(yg, 5)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return str(self.xr) + "," + str(self.yr) + "," + str(self.u) + "," + str(self.r) + "," + str(self.d) + "," + str(self.l) + "," + str(self.xg) + "," + str(self.yg)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, FlatState) and self.xr == other.xr and self.yr == other.yr and self.u == other.u and self.r == other.r and self.d == other.d and self.l == other.l and self.xg == other.xg and self.yg == other.yg
