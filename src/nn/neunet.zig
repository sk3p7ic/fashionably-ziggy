const std = @import("std");
const nn_layers = @import("layers.zig");
const nn_funcs = @import("ufuncs.zig");
const matrices = @import("../utils/matrices.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const LinearLayer = nn_layers.LinearLayer;
const Matrix = matrices.Matrix;

pub fn NNForwardResult(comptime T: type) type {
    return struct {
        const Self = @This();

        Z1: Matrix(T),
        A1: Matrix(T),
        Z2: Matrix(T),
        A2: Matrix(T),

        pub fn init(
            Z1: Matrix(T),
            A1: Matrix(T),
            Z2: Matrix(T),
            A2: Matrix(T),
        ) !Self {
            return Self{ .Z1 = try Z1.subdivide(Z1.rows, Z1.cols, 0, 0), .A1 = try A1.subdivide(A1.rows, A1.cols, 0, 0), .Z2 = try Z2.subdivide(Z2.rows, Z2.cols, 0, 0), .A2 = try A2.subdivide(A2.rows, A2.cols, 0, 0) };
        }

        pub fn initEmpty() Self {
            return Self{ .Z1 = null, .A1 = null, .Z2 = null, .A2 = null };
        }
    };
}

pub fn NN(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            // TODO: Maybe allow user to pass in arbitrary num of layers?
            return Self{ .allocator = allocator };
        }

        pub fn deinit(self: Self) void {
            for (self.modules.items) |layer| {
                layer.deinit();
            }
            self.modules.deinit();
        }

        pub fn forward(self: Self, A: Matrix(T)) !NNForwardResult(T) {
            const Lyr1 = try LinearLayer(T, nn_funcs.relu_activation).init(86, 784, self.allocator);
            const Lyr2 = try LinearLayer(T, nn_funcs.softmax_activation).init(10, 86, self.allocator);
            const layer1 = try Lyr1.forward(A);
            defer layer1.deinit();
            const layer2 = try Lyr2.forward(layer1.A);
            defer layer2.deinit();
            // Copy the results to a memory-safe struct
            const epoch_result = try NNForwardResult(T).init(layer1.Z, layer1.A, layer2.Z, layer2.A);
            return epoch_result;
        }
    };
}
