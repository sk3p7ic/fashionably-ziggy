const std = @import("std");
const matrices = @import("../utils/matrices.zig");
const ufuncs = @import("ufuncs.zig");
const Allocator = std.mem.Allocator;
const Matrix = matrices.Matrix;

pub fn LinearLayer(comptime T: type, comptime F: fn (comptime type, anytype) type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        weights: Matrix(T),
        biases: Matrix(T),

        pub fn init(n_neurons: usize, n_features: usize, allocator: Allocator) !Self {
            const layer = Self{
                .allocator = allocator,
                .weights = try Matrix(T).init(n_neurons, n_features, allocator),
                .biases = try Matrix(T).init(n_neurons, 1, allocator),
            };
            layer.weights.frand(0);
            layer.biases.fill(0);
            return layer;
        }

        pub fn deinit(self: Self) void {
            self.weights.deinit();
            self.biases.deinit();
        }

        pub fn forward(self: Self, A: Matrix(T)) !Matrix(T) {
            if (A.rows != self.weights.cols) {
                return matrices.MatrixError.ShapeMismatch;
            }
            const p = try self.weights.dot(A);
            defer p.deinit();
            const s = try p.sumWithVec(self.biases);
            defer s.deinit();
            for (0..s.rows) |r| {
                for (0..s.cols) |c| {
                    const ptr = try s.at(r, c);
                    const act = F(T, ptr.*);
                    ptr.* = act;
                }
            }
            return s;
        }
    };
}
