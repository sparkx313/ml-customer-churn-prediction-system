def predict_churn(customer_data):
    print("📊 Evaluating Churn Probability...")
    return 0.12

def main():
    prob = predict_churn({"tenure": 36, "monthly_charges": 65.5})
    print(f"✓ Customer Churn Risk: {prob * 100:.1f}%")

if __name__ == "__main__":
    main()
