balance = 1000
pin = "1234"

entered_pin = input("Enter your 4-digit PIN: ")

if entered_pin == pin:
    print("Login successful!\n")

    while True:
        print("=== ATM Menu ===")
        print("1. Check Balance")
        print("2. Deposit Money")
        print("3. Withdraw Money")
        print("4. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            print(f"Your balance is ₹{balance}\n")

        elif choice == "2":
            amount = float(input("Enter amount to deposit: ₹"))
            balance += amount
            print(f"₹{amount} deposited. New balance: ₹{balance}\n")

        elif choice == "3":
            amount = float(input("Enter amount to withdraw: ₹"))
            if amount <= balance:
                balance -= amount
                print(f"₹{amount} withdrawn. New balance: ₹{balance}\n")
            else:
                print("Not enough balance.\n")

        elif choice == "4":
            print("Thanks for using our ATM. Goodbye!")
            break

        else:
            print("Invalid option. Try again.\n")
else:
    print("Wrong PIN. Access denied.")