"""Links between agents"""


class Link:
    def __init__(self, beg, end):
        self.own = [beg, end]
        self.ids = [beg.id, end.id]
