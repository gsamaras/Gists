# -*- coding: utf-8 -*-

'''
1. Visit Instagram from laptop's browser, get followers and accounts you follow from Developer Tools (you need to scroll down till the last account)
2. Put in editor and replace all single quotes with double quotes.
3. Put them in the respective 'following_str' and 'followers_str' strings and execute the Python script.
'''

# Find accounts you follow
following_str = '<div </div>'
split_str = following_str.split("title")
print(len(split_str))
#print(split_str[0])
#print(split_str[1])
following_n = len(split_str)
following_lst = []
for i in range(1, following_n):
    #print(split_str[i].split('="', 1)[1].split('"', 1)[0])
    following_lst.append(split_str[i].split('="', 1)[1].split('"', 1)[0])
#print(split_str[1].split('="', 1)[1].split('"', 1)[0])
print(len(following_lst))

# Find accounts that follow you
followers_str = '<div </div>'
split_str = followers_str.split("title")
print(len(split_str))
#print(split_str[0])
#print(split_str[1])
followers_n = len(split_str)
followers_lst = []
for i in range(1, followers_n):
    #print(split_str[i].split('="', 1)[1].split('"', 1)[0])
    followers_lst.append(split_str[i].split('="', 1)[1].split('"', 1)[0])
#print(split_str[1].split('="', 1)[1].split('"', 1)[0])
print(len(followers_lst))

# Find which accounts don't follow you back
for following in following_lst:
    if following not in followers_lst:
        print(following)