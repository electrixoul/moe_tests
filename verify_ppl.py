#!/usr/bin/env python3
"""Verify perplexity calculation is correct."""

import math

print("="*60)
print("Perplexity Calculation Verification")
print("="*60)

# From the training output
test_cases = [
    (4.3073, 74.24),  # Step 1
    (2.4580, 11.68),  # Step 201
    (2.2258, 9.26),   # Step 401
    (1.7723, 5.88),   # Step 601
    (1.5202, 4.57),   # Step 801
    (1.4409, 4.22),   # Step 1001
    (1.3989, 4.05),   # Step 1201
    (1.3171, 3.73),   # Step 1401
    (1.2479, 3.48),   # Step 1601
]

print("\nVerifying PPL = exp(CE):\n")
print(f"{'CE Loss':<10} {'Reported PPL':<15} {'Calculated PPL':<15} {'Match?'}")
print("-" * 60)

all_correct = True
for ce, reported_ppl in test_cases:
    calculated_ppl = math.exp(ce)
    diff = abs(calculated_ppl - reported_ppl)
    match = diff < 0.1  # Allow small floating point differences
    all_correct = all_correct and match
    status = "✓" if match else "✗"
    print(f"{ce:<10.4f} {reported_ppl:<15.2f} {calculated_ppl:<15.2f} {status}")

print("\n" + "="*60)
if all_correct:
    print("✓ All perplexity calculations are CORRECT!")
else:
    print("✗ Some calculations are incorrect")

print("\nExplanation:")
print("-" * 60)
print("For character-level language modeling on Tiny Shakespeare:")
print(f"  - Vocabulary size: 65 characters")
print(f"  - Random guessing baseline: PPL ≈ 65")
print(f"  - Perfect model: PPL = 1")
print(f"  - Current model (step 1601): PPL ≈ 3.48")
print()
print("A PPL of 3-4 means the model is effectively choosing among")
print("3-4 likely characters on average, which is GOOD performance!")
print("="*60)
