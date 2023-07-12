const std = @import("std");
const matrices = @import("../utils/matrices.zig");
const Allocator = std.mem.Allocator;
const Matrix = matrices.Matrix;

pub fn LinearLayer(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        weights: Matrix(T),
        biases: Matrix(T),

        pub fn init(n_neurons: usize, n_features: usize, allocator: Allocator) !Self {
            const layer = Self{ .allocator = allocator, .weights = try Matrix(T).init(n_neurons, n_features, allocator), .biases = try Matrix(T).init(n_neurons, 1, allocator) };
            layer.weights.frand(0);
            layer.biases.fill(0);
            return layer;
        }

        pub fn deinit(self: Self) void {
            self.weights.deinit();
        }
    };
}