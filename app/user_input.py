import json

# Load predefined test cases
def load_test_cases(filename="test/tests_input.json"):
    with open(filename, "r") as f:
        return json.load(f)

# Get user input or use test case
def get_user_input():
    test_cases = load_test_cases()
    
    user_choice = input("Enter 1 to provide custom input or press Enter to use a predefined test case: ").strip()

    if user_choice == "1":
        software_category = input("Enter the software category: ").strip()
        capabilities = [cap.strip() for cap in input("Enter capabilities (comma-separated): ").split(",")]
    else:
        print("\nAvailable test cases:")
        for i, test in enumerate(test_cases, start=1):
            print(f"{i}. {test['software_category']} - {test['capabilities']}")

        test_choice = input("\nEnter test case number (or press Enter for the first one): ").strip()
        test_case = test_cases[int(test_choice) - 1] if test_choice.isdigit() and 1 <= int(test_choice) <= len(test_cases) else test_cases[0]

        software_category = test_case["software_category"]
        capabilities = test_case["capabilities"]

    return software_category, capabilities

# # Load input
# software_category, capabilities = get_user_input()

# # Print final values
# print("\nSelected Input:")
# print(f"Software Category: {software_category}")
# print(f"Capabilities: {capabilities}")
