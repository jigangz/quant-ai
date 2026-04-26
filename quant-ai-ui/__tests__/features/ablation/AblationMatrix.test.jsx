import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AblationMatrix from "@/features/ablation/AblationMatrix";

const MATRIX = {
  direction: {
    "ta_basic":              { model_id: "x", auc: 0.523, f1: 0.34 },
    "ta_basic + sentiment":  { model_id: "y", auc: 0.591, f1: 0.42, delta_auc: 0.068 },
  },
  volatility: {
    "ta_basic":              { model_id: "x", qlike: 0.171, r2: 0.019 },
    "ta_basic + sentiment":  { model_id: "y", qlike: 0.142, r2: 0.064, delta_qlike: -0.029 },
  },
};

describe("AblationMatrix", () => {
  it("renders cells for every (target, feature_set)", () => {
    render(<AblationMatrix matrix={MATRIX} />);
    expect(screen.getByText(/0\.523/)).toBeInTheDocument();
    expect(screen.getByText(/0\.591/)).toBeInTheDocument();
    expect(screen.getByText(/0\.171/)).toBeInTheDocument();
    expect(screen.getByText(/0\.142/)).toBeInTheDocument();
  });
});
