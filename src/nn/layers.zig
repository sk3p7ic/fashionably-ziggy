const std = @import("std");
const matrices = @import("../utils/matrices.zig");
const ufuncs = @import("ufuncs.zig");
const Allocator = std.mem.Allocator;
const Matrix = matrices.Matrix;

pub fn ForwardResult(comptime T: type) type {
    return struct {
        const Self = @This();

        A: Matrix(T),
        Z: Matrix(T),

        pub fn init(A: Matrix(T), Z: Matrix(T)) !Self {
            return Self{ .A = try A.subdivide(A.rows, A.cols, 0, 0), .Z = try Z.subdivide(Z.rows, Z.cols, 0, 0) };
        }

        pub fn deinit(self: Self) void {
            self.A.deinit();
            self.Z.deinit();
        }
    };
}

pub fn LinearLayer(comptime T: type, comptime F: fn (comptime type, anytype) void) type {
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

        pub fn forward(self: Self, A: Matrix(T)) !ForwardResult(T) {
            if (A.rows != self.weights.cols) {
                return matrices.MatrixError.ShapeMismatch;
            }
            const p = try self.weights.dot(A);
            defer p.deinit();
            const s = try p.sumWithVec(self.biases);
            const a = try s.subdivide(s.rows, s.cols, 0, 0); // Copy s
            defer s.deinit();
            defer a.deinit();
            F(T, &a); // Activation function
            return try ForwardResult(T).init(a, s);
        }
    };
}
