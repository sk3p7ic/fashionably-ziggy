const std = @import("std");
const matrices = @import("../utils/matrices.zig");
const Matrix = matrices.Matrix;

pub fn relu_activation(comptime T: type, M: *Matrix(T)) void {
    for (0..M.rows) |r| {
        for (0..M.cols) |c| {
            const ptr = try M.at(r, c);
            if (ptr.* == 0) {
                ptr.* = 0;
            }
        }
    }
}

pub fn softmax_activation(comptime T: type, M: *Matrix(T)) void {
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

pub const ActivationFunctionTag = enum { relu, softmax };

pub const ActivationFunction = union(ActivationFunctionTag) { relu: @TypeOf(relu_activation), softmax: @TypeOf(softmax_activation) };

pub fn get_activation_fn(afu: ActivationFunction) type {
    return switch (afu) {
        ActivationFunctionTag.relu => |f| f,
        ActivationFunctionTag.softmax => |f| f,
    };
}
