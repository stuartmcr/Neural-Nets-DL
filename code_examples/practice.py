# solving for square root of x, expected value is 5
x = 25
epsilon = 0.01
# start counter at 0
numGuesses = 0

low = 0.0  # lowest possible value
high = max(1.0, x)  # prevents high value dropping below 1
ans = (high + low) / 2.0  # average

while abs(ans ** 2 - x) >= epsilon:  # run until desired level of accuracy is met
    print("low =", low, "high =", high, "ans =", ans)  # print current values
    numGuesses += 1  # adds 1 to counter for each iteration
    if ans ** 2 < x:
        low = ans  # brings lower bound up
    else:
        high = ans  # brings upper bound down
    ans = (high + low) / 2.0  # calcuates new ans value

# once level of accuracy (epsilon) is met, snaps out of while loop
print("numGuesses =", numGuesses)
print(ans, "is close to square root of", x)
