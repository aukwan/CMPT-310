"""
The lectures are a little dry. Talking about searches are helpful to me as I struggled
with them in past courses. Looking forward to learning more about AI.
Around 5 hours spent on this assignment

To run this script:
	python EvenOdd.py

In order to pass the autograder your function should
return a list of even numbers given any list of integers.
If you run the above script, a correct function should return:

Even numbers are [2, 4]
"""

def getEvenNumbers(numbers):
        evens = []
        "*** Add your code in here ***"
        num = 0

        while (num < len(numbers)):
                if numbers[num] % 2 == 0:
                        evens.append(numbers[num])
                num+=1
        return evens


if __name__ == '__main__':
	myList = [1, 2, 3, 4, 5]
	print("Even numbers are {}".format(getEvenNumbers(myList)))
