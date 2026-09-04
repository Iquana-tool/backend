import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import EditableLabels from "./EditableLabels";

const moveLabel = vi.fn();
const fetchLabels = vi.fn();
const fetchLabelNestingSummary = vi.fn();

vi.mock("../../../api", () => ({
    moveLabel: (...args) => moveLabel(...args),
    fetchLabels: (...args) => fetchLabels(...args),
    fetchLabelNestingSummary: (...args) => fetchLabelNestingSummary(...args),
    createLabel: vi.fn(),
    updateLabel: vi.fn(),
    deleteLabel: vi.fn(),
}));

// jsdom has no drag implementation; the handlers only need these two members.
const dataTransfer = () => ({ setData: vi.fn(), effectAllowed: "", dropEffect: "" });

const rowFor = (name) =>
    screen.getByTitle(`Drag "${name}" onto another label to make it a part of it`);

// cell > nucleus, with tissue as a second whole object to move nucleus under.
const LABELS = [
    { id: 1, name: "cell", parent_id: null, value: 1 },
    { id: 2, name: "nucleus", parent_id: 1, value: 2 },
    { id: 3, name: "tissue", parent_id: null, value: 3 },
];

const dataset = { id: 7, name: "Test Dataset" };

const openMoveDialogForNucleus = () => {
    fireEvent.click(
        screen.getByTitle('Move "nucleus" — make it a part of something else')
    );
};

describe("EditableLabels — moving a label", () => {
    beforeEach(() => {
        moveLabel.mockReset();
        fetchLabels.mockReset();
        fetchLabelNestingSummary.mockReset();
        fetchLabels.mockResolvedValue(LABELS);
        fetchLabelNestingSummary.mockResolvedValue({
            success: true,
            summary: { nested_total: 0, by_container_label: {} },
        });
    });

    it("shows the tree expanded so nested labels are visible without interaction", () => {
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);

        // `nucleus` is nested one level down and must be on screen straight away.
        expect(screen.getByText("nucleus")).toBeInTheDocument();
        expect(screen.getByText("1 part")).toBeInTheDocument();
    });

    it("offers every label except the moved one and its own parts as a destination", () => {
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);
        fireEvent.click(screen.getByTitle('Move "cell" — make it a part of something else'));

        const options = Array.from(screen.getByLabelText("Make it a part of").options).map(
            (option) => option.textContent
        );

        expect(options).toContain("tissue");
        // `cell` itself and its part `nucleus` would both form a cycle.
        expect(options).not.toContain("cell");
        expect(options).not.toContain("cell › nucleus");
    });

    it("asks for confirmation before detaching objects a move would strand", async () => {
        moveLabel.mockResolvedValueOnce({
            blocked: true,
            message: "2 annotated object(s) would be invalidated by this move.",
            affectedCount: 2,
            affectedObjects: [{ contour_id: 11, image_id: 3 }],
        });
        moveLabel.mockResolvedValueOnce({ success: true, detached_count: 2 });

        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);
        openMoveDialogForNucleus();

        fireEvent.change(screen.getByLabelText("Make it a part of"), { target: { value: "3" } });
        fireEvent.click(screen.getByRole("button", { name: "Move" }));

        // First attempt reports the damage instead of doing it.
        expect(await screen.findByText(/would be left in a place this move makes illegal/i))
            .toBeInTheDocument();
        expect(moveLabel).toHaveBeenCalledWith(2, 3, { detachAffected: false });

        // Only after the user accepts is the destructive variant sent.
        fireEvent.click(screen.getByRole("button", { name: /Detach 2 objects and move/i }));

        await waitFor(() =>
            expect(moveLabel).toHaveBeenLastCalledWith(2, 3, { detachAffected: true })
        );
        await waitFor(() => expect(fetchLabels).toHaveBeenCalled());
    });

    it("sends null as the destination when moving a label to the top level", async () => {
        moveLabel.mockResolvedValue({ success: true, detached_count: 0 });

        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);
        openMoveDialogForNucleus();

        fireEvent.change(screen.getByLabelText("Make it a part of"), { target: { value: "root" } });
        fireEvent.click(screen.getByRole("button", { name: "Move" }));

        await waitFor(() =>
            expect(moveLabel).toHaveBeenCalledWith(2, null, { detachAffected: false })
        );
    });

    it("disables the move when the chosen destination is where the label already is", () => {
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);
        openMoveDialogForNucleus();

        // The dialog opens on the label's current parent, so there is nothing to do yet.
        expect(screen.getByRole("button", { name: "Move" })).toBeDisabled();

        fireEvent.change(screen.getByLabelText("Make it a part of"), { target: { value: "3" } });
        expect(screen.getByRole("button", { name: "Move" })).toBeEnabled();
    });
});


describe("EditableLabels — dragging a label onto another", () => {
    beforeEach(() => {
        moveLabel.mockReset();
        fetchLabels.mockReset();
        fetchLabelNestingSummary.mockReset();
        fetchLabels.mockResolvedValue(LABELS);
        moveLabel.mockResolvedValue({ success: true, detached_count: 0 });
    });

    const dragOnto = (sourceName, targetName) => {
        fireEvent.dragStart(rowFor(sourceName), { dataTransfer: dataTransfer() });
        fireEvent.dragOver(rowFor(targetName), { dataTransfer: dataTransfer() });
        fireEvent.drop(rowFor(targetName), { dataTransfer: dataTransfer() });
    };

    it("moves the label under the row it is dropped on", async () => {
        fetchLabelNestingSummary.mockResolvedValue({
            success: true,
            summary: { nested_total: 0, by_container_label: {} },
        });
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);

        dragOnto("nucleus", "tissue");

        await waitFor(() =>
            expect(moveLabel).toHaveBeenCalledWith(2, 3, { detachAffected: false })
        );
    });

    it("warns with a live count while hovering a destination that would strand objects", async () => {
        // Three `nucleus` objects, all currently inside a `cell`: moving the label
        // anywhere but under `cell` strands all three.
        fetchLabelNestingSummary.mockResolvedValue({
            success: true,
            summary: { nested_total: 3, by_container_label: { 1: 3 } },
        });
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);

        fireEvent.dragStart(rowFor("nucleus"), { dataTransfer: dataTransfer() });
        await waitFor(() => expect(fetchLabelNestingSummary).toHaveBeenCalledWith(2));
        fireEvent.dragOver(rowFor("tissue"), { dataTransfer: dataTransfer() });

        expect(await screen.findByText("would detach 3 objects")).toBeInTheDocument();
        // Hovering is not committing: nothing has been asked of the server yet.
        expect(moveLabel).not.toHaveBeenCalled();
    });

    it("refuses to drop a label onto one of its own parts", async () => {
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);

        dragOnto("cell", "nucleus");

        await waitFor(() => expect(fetchLabelNestingSummary).toHaveBeenCalled());
        expect(moveLabel).not.toHaveBeenCalled();
    });

    it("offers a drop target that takes a label back out of the hierarchy", async () => {
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);

        fireEvent.dragStart(rowFor("nucleus"), { dataTransfer: dataTransfer() });

        const strip = await screen.findByText(
            /Drop here to make "nucleus" a whole object of its own/
        );
        fireEvent.dragOver(strip, { dataTransfer: dataTransfer() });
        fireEvent.drop(strip, { dataTransfer: dataTransfer() });

        await waitFor(() =>
            expect(moveLabel).toHaveBeenCalledWith(2, null, { detachAffected: false })
        );
    });

    it("opens the confirmation when the server refuses a dropped move", async () => {
        fetchLabelNestingSummary.mockResolvedValue({
            success: true,
            summary: { nested_total: 2, by_container_label: { 1: 2 } },
        });
        moveLabel.mockResolvedValueOnce({
            blocked: true,
            message: "2 annotated object(s) would be invalidated by this move.",
            affectedCount: 2,
            affectedObjects: [],
        });
        render(<EditableLabels dataset={dataset} labels={LABELS} onLabelsUpdated={vi.fn()} />);

        dragOnto("nucleus", "tissue");

        expect(await screen.findByText(/would be left in a place this move makes illegal/i))
            .toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Detach 2 objects and move/i })
        ).toBeInTheDocument();
    });
});
