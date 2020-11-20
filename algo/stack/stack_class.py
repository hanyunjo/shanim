class Stack:
    def __inti__(self):
        self.item = []

    def push(self, item):
        self.item.append(item)
    
    def pop(self):
        if self.isEmpty():
            return 
        else:
            return self.item.pop()
    
    def isEmpty(self):
        return not self.item