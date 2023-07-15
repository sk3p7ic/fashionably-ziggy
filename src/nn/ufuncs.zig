const std = @import("std");
const matrices = @import("../utils/matrices.zig");
const Matrix = matrices.Matrix;

pub fn relu_activation(comptime T: type, M: *const Matrix(T)) matrices.MatrixError!void {
    for (0..M.rows) |r| {
        for (0..M.cols) |c| {
            const ptr = try M.at(r, c);
            if (ptr.* == 0) {
                ptr.* = 0;
            }
        }
    }
}

pub fn softmax_activation(comptime T: type, M: *const Matrix(T)) matrices.MatrixError!void {
    var m_sum: T = 0;
    for (M.items) |e| {
        m_sum += e;
    }
    for (0..M.rows) |r| {
        for (0..M.cols) |c| {
            const ptr = try M.at(r, c);
            const exp = std.math.exp(ptr.*);
            ptr.* = exp / m_sum;
        }
    }
}
