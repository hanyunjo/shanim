class Game:
    title = ''
    comp = ''
    grade = ''
    price = ''

    def __init__(self, tag):
        self.title= self.get_text(tag, 'div.WsMG1c')
        self.comp = self.get_text(tag, 'div.KoLSrc')
        #comp = c.select('div.KoLSrc')[0].get('title'), This html don't exited title
        #self.grade = self.get_text(tag, 'div.pf5lIe')
        self.price = self.get_text(tag, 'div.LCATme')
        #price = c.select('span.VFPpfd')[0].text, price = VfPpfd ZdBevf i5DZme / LCATme
        #WsMG1c=title, KoLSrc = company

    def get_text(self, parent_tag, selector):
        t = self.get_tag(parent_tag, selector)
        return "" if t == None else t.text.strip()

    def get_attr(self, parent_tag, selector, attr):
        at = self.get_tag(parent_tag, selector)
        return "" if at == None else at.get(attr).strip()

    def get_tag(self, parent_tag, selector):
        tag = parent_tag.select(selector)
        return "" if tag == None or len(tag) == 0 else tag[0]

    def __str__(self):
        return self.to_str()

    def to_str(self):
        return "Title : {}, Company : {}, Price : {}".format(self.title, self.comp, self.price)
