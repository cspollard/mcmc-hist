
module Outline where

open import Data.Vec.Recursive using ([]; foldl; zipWith; _^_)
open import Algebra using (Monoid)
open import Level using (0ℓ)
open import Data.Nat using (ℕ)
open import Function using (id)

postulate
  ℝ-monoid : Monoid 0ℓ 0ℓ


open Monoid ℝ-monoid
  using (_∙_)
  renaming (Carrier to ℝ; ε to 1#)

Likelihood : ∀ {a} → Set a → Set a
Likelihood A = A → ℝ

Pred : ℕ → Set
Pred n = ℝ ^ n

Data : ℕ → Set
Data n = ℕ ^ n

postulate
  poisson : ℝ → ℕ → ℝ

llh : ∀ {n} → Pred n → Data n → ℝ
llh {n} p d = foldl (λ _ → ℝ) 1# id (λ _ → _∙_) n (zipWith poisson n p d)

