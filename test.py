



def wrap(string, max_width):
    word_list=[]
    word=''
    for i in range(1,len(string)+1):
        word+=string[i-1]
        if i%4==0:
            word_list.append(word)
            word=''  
        elif i==len(string):
            word_list.append(word)
            
      
    return word_list

def join_str(string,count):
    full_str=''
    for i in range(count):
        full_str+=string
    
    return full_str


print(join_str('.|.',3))