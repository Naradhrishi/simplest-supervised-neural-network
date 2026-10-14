# 1. Setup the Dataset (Supervised Learning)
x = 2.0   # Input (The Question)
y = 10.0  # Target (The Correct Answer)

# 2. Initialize the Adjustable Parameters (Our Starting State)
w = 3.0   # Initial Weight
b = 1.0   # Initial Bias

# 3. Hyperparameters
learning_rate = 0.1
iterations = 40  # We set this to 40 so I can watch it hit perfection

print(f"Starting Training... Goal: Input {x} -> Target {y}\n")
print(f"{'Iteration':<10} | {'Weight (w)':<12} | {'Bias (b)':<12} | {'Prediction (ŷ)':<16} | {'Error (E)':<12}")
print("-" * 75)

# 4. The Training Loop
for i in range(1, iterations + 1):

    # --- PHASE 1: The Forward Pass ---
    y_hat = (x * w) + b
    error = 0.5 * (y_hat - y) ** 2
    
    # Print the current state BEFORE we update (matching our hand calculations)
    # We use formatting to show up to 17 decimal places to catch the precision drop
    if i <= 7 or i >= 20:
        print(f"Iter {i:<5} | {w:<12.6f} | {b:<12.6f} | {y_hat:<16.12f} | {error:<12.17f}")
    elif i == 8:
        print("... [Skipping intermediate logs for space] ...")

    # --- PHASE 2: The Backward Pass (Manual Calculus Chain Rule) ---
    # dE/dy_hat (Outside derivative)
    de_dyhat = y_hat - y  
    
    # dy_hat/dw = x, and dy_hat/db = 1 (Inside derivatives)
    de_dw = de_dyhat * x   # Total Chain Rule for Weight
    de_db = de_dyhat * 1   # Total Chain Rule for Bias
    
    # --- PHASE 3: Gradient Descent Update ---
    w = w - (learning_rate * de_dw)
    b = b - (learning_rate * de_db)


print("-" * 75)
print(f"Final Optimized State: w = {w:.6f}, b = {b:.6f}")