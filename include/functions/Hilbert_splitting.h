#ifndef NDG_FUNCTIONS_HILBERT_SPLITTING_H
#define NDG_FUNCTIONS_HILBERT_SPLITTING_H

#include <array>

namespace SEM { namespace Host { namespace Hilbert {
    /**
     * @brief Describes the geometrical arrangement of a cell, one of four possible values.
     * 
     * This is necessary to the table-driven algorithm for the Hilbert curve, where the order
     * and status oh the next level cells is determined by its parent's status.
     */
    enum Status : int {H, A, R, B};

    /**
     * @brief Returns the geometric status of the first element of a curve depending on which side the curve exits the element.
     * 
     * @param outgoing_side Side through which the curve exits the element, which is the line from this element to the next.
     * @return Status Geometric arrangement of the cell.
     */
    auto deduct_first_element_status(size_t outgoing_side) -> Status;

    /**
     * @brief Returns the geometric status of the last element of a curve depending on which side the curve enters the element.
     * 
     * @param incoming_side Side through which the curve enters the element, which is the line from the previous element to this one.
     * @return Status Geometric arrangement of the cell.
     */
    auto deduct_last_element_status(size_t incoming_side) -> Status;

    /**
     * @brief Returns the geometric status of an element depending on which side the curve enters and exits the element.
     * 
     * @param incoming_side Side through which the curve enters the element, which is the line from the previous element to this one.
     * @param outgoing_side Side through which the curve exits the element, which is the line from this element to the next.
     * @return Status Geometric arrangement of the cell.
     */
    auto deduct_element_status(size_t incoming_side, size_t outgoing_side) -> Status;

    /**
     * @brief Returns the order in which the child elements should be placed in the curve depending on the parent's status.
     * 
     * The returned child indices are from the bottom left, counter-clockwise. This means that if {2, 3, 0, 1} is returned,
     * this is the order the child elements should be connected: top right, top left, bottom left, bottom right.
     * 
     * @param parent_status Geometric arrangement of the parent.
     * @param rotation Which side is the element's first side.
     * @return std::array<size_t, 4> Order of the child elements in the curve, numbered from the bottom left counter-clockwise.
     */
    auto child_order(Status parent_status, int rotation) -> std::array<size_t, 4>;

    /**
     * @brief Returns the statuses of the child elements depending on the parent's status.
     * 
     * The returned child statuses are from the bottom left, counter-clockwise. This means that if {A, B, H, H} is returned,
     * the bottom left child will be A, the bottom right child will be B, the top right will be H, and the top left will be H.
     * 
     * @param parent_status Geometric arrangement of the parent.
     * @param rotation Which side is the element's first side.
     * @return std::array<Status, 4> Statuses of the child elements in the curve, numbered from the bottom left counter-clockwise.
     */
    auto child_statuses(Status parent_status, int rotation) -> std::array<Status, 4>;
}}}

#endif