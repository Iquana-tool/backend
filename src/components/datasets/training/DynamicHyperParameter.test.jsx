import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import DynamicHyperParameter, { coerceValue } from "./DynamicHyperParameter";

describe("DynamicHyperParameter", () => {
  it("coerces values correctly for each type including false booleans", () => {
    expect(coerceValue("123", "int")).toBe(123);
    expect(coerceValue("", "int")).toBe("");
    expect(coerceValue("3.14", "float")).toBeCloseTo(3.14);
    expect(coerceValue("", "float")).toBe("");
    expect(coerceValue(true, "bool")).toBe(true);
    expect(coerceValue(false, "bool")).toBe(false);
    expect(coerceValue("true", "bool")).toBe(true);
    expect(coerceValue("false", "bool")).toBe(false);
    expect(coerceValue("hello", "str")).toBe("hello");
  });

  it("handles boolean options dropdown correctly selecting false", () => {
    const onChange = jest.fn();
    const param = {
      key: "normalize",
      label: "Normalize",
      type: "bool",
      options: [true, false],
      default_value: true,
    };

    render(<DynamicHyperParameter param={param} value={true} onChange={onChange} />);

    const select = screen.getByRole("combobox", { name: /normalize/i });
    expect(select.value).toBe("true");

    fireEvent.change(select, { target: { value: "false" } });
    expect(onChange).toHaveBeenCalledWith("normalize", false);
  });

  it("scopes control and label IDs using idPrefix", () => {
    const param = {
      key: "threshold",
      label: "Score Threshold",
      type: "float",
      min_value: 0.0,
      max_value: 1.0,
      default_value: 0.5,
    };

    render(
      <DynamicHyperParameter
        param={param}
        value={0.5}
        onChange={jest.fn()}
        idPrefix="label-42"
        compact
      />
    );

    const slider = screen.getByRole("slider", { name: /score threshold/i });
    expect(slider.id).toBe("label-42-param-threshold");
  });

  it("renders description in title attribute in compact mode", () => {
    const param = {
      key: "min_target_frac",
      label: "Min Target Fraction",
      type: "float",
      default_value: 0.0001,
      description: "Minimum size of predicted target object relative to image",
    };

    render(
      <DynamicHyperParameter
        param={param}
        value={0.0001}
        onChange={jest.fn()}
        compact
      />
    );

    const label = screen.getByText("Min Target Fraction");
    expect(label).not.toHaveAttribute("title");

    const info = screen.getByRole("img", {
      name: "About Min Target Fraction: Minimum size of predicted target object relative to image",
    });
    expect(info).not.toHaveAttribute("title");
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "Minimum size of predicted target object relative to image"
    );
  });

  it("renders a select dropdown when options are provided", () => {
    const onChange = jest.fn();
    const param = {
      key: "arch",
      label: "Architecture",
      type: "str",
      options: ["resnet50", "vit_b", "vit_h"],
      default_value: "vit_b",
    };

    render(<DynamicHyperParameter param={param} value="resnet50" onChange={onChange} />);

    const select = screen.getByRole("combobox", { name: /architecture/i });
    expect(select.value).toBe("resnet50");

    fireEvent.change(select, { target: { value: "vit_h" } });
    expect(onChange).toHaveBeenCalledWith("arch", "vit_h");
  });

  it("renders a checkbox when type is bool", () => {
    const onChange = jest.fn();
    const param = {
      key: "use_flip",
      label: "Use Horizontal Flip",
      type: "bool",
      default_value: false,
    };

    render(<DynamicHyperParameter param={param} value={false} onChange={onChange} />);

    const checkbox = screen.getByRole("checkbox", { name: /use horizontal flip/i });
    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith("use_flip", true);
  });

  it("renders a slider when min_value and max_value are finite and no options", () => {
    const onChange = jest.fn();
    const param = {
      key: "threshold",
      label: "Score Threshold",
      type: "float",
      min_value: 0.0,
      max_value: 1.0,
      step: 0.05,
      default_value: 0.5,
    };

    render(<DynamicHyperParameter param={param} value={0.7} onChange={onChange} />);

    const slider = screen.getByRole("slider", { name: /score threshold/i });
    expect(slider.value).toBe("0.7");

    fireEvent.change(slider, { target: { value: "0.85" } });
    expect(onChange).toHaveBeenCalledWith("threshold", 0.85);
  });

  it("renders a number input when unbounded numeric type", () => {
    const onChange = jest.fn();
    const param = {
      key: "max_iter",
      label: "Max Iterations",
      type: "int",
      default_value: 100,
    };

    render(<DynamicHyperParameter param={param} value={150} onChange={onChange} />);

    const input = screen.getByRole("spinbutton", { name: /max iterations/i });
    expect(input.value).toBe("150");

    fireEvent.change(input, { target: { value: "200" } });
    expect(onChange).toHaveBeenCalledWith("max_iter", 200);
  });

  it("renders a text input for string type without options", () => {
    const onChange = jest.fn();
    const param = {
      key: "tag",
      label: "Custom Tag",
      type: "str",
      default_value: "prod",
    };

    render(<DynamicHyperParameter param={param} value="test" onChange={onChange} />);

    const input = screen.getByRole("textbox", { name: /custom tag/i });
    expect(input.value).toBe("test");

    fireEvent.change(input, { target: { value: "staging" } });
    expect(onChange).toHaveBeenCalledWith("tag", "staging");
  });

  it("renders description text inline and in the info tooltip", () => {
    const param = {
      key: "lr",
      label: "Learning Rate",
      type: "float",
      default_value: 0.001,
      description: "Initial learning rate for optimizer",
    };

    render(<DynamicHyperParameter param={param} value={0.001} onChange={jest.fn()} />);

    expect(screen.getAllByText("Initial learning rate for optimizer")).toHaveLength(2);
  });

  it("disables input controls when disabled is true", () => {
    const param = {
      key: "threshold",
      label: "Score Threshold",
      type: "float",
      min_value: 0.0,
      max_value: 1.0,
      default_value: 0.5,
    };

    render(<DynamicHyperParameter param={param} value={0.5} onChange={jest.fn()} disabled={true} />);

    const slider = screen.getByRole("slider", { name: /score threshold/i });
    expect(slider).toBeDisabled();
  });
});
