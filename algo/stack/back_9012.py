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


num = int(input())
stack = Stack()

for i in range(num):
    line = input()
    result = 1

    for l in range(len(line)):
        if (line[l] == '{') or (line[l] == '[') or (line[l] == '('):
            stack.push(line[l])
        else:
            if line[l] == '}':
                if stack.pop() != '{':
                    print('NO')
                    result = 0
                    break
            elif line[l] == ']':
                if stack.pop() != '[':
                    print('NO')
                    result = 0
                    break
            else:
                if stack.pop() != '(':
                    print('NO')
                    result = 0
                    break
        
    if result == '1' and stack.isEmpty():
        print('YES')
    else:
        print('NO')
        
    



