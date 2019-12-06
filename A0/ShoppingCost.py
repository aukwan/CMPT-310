"""
To run this script:
        python shoppingCost.py

If you run the above script, a correct calculateShoppingCost function should return:

The final cost for our shopping cart is 35.58
"""

import csv

def calculateShoppingCost(productPrices, shoppingCart):
        finalCost = 0
        "*** Add your code in here ***"
        for item in shoppingCart:
                finalCost = finalCost + (float(shoppingCart[item]) * float(productPrices[item]))
        return finalCost


def createPricesDict(filename):
        productPrice = {}
        "*** Add your code in here ***"
        inputFile = csv.reader(open(filename, 'r'))
        for row in inputFile:
                i,j = row
                productPrice[i] = j
        return productPrice


if __name__ == '__main__':
        prices = createPricesDict("Grocery.csv")
        myCart = {"Bacon": 2,
                      "Homogenized milk": 1,
                      "Eggs": 5}
        print("The final cost for our shopping cart is {}".format(calculateShoppingCost(prices, myCart)))
        
