const std = @import("std");
const nn_layers = @import("layers.zig");
const matrices = @import("../utils/matrices.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const LinearLayer = nn_layers.LinearLayer;
const Matrix = matrices.Matrix;

pub fn NN(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        modules: ArrayList(LinearLayer(T)),

        pub fn init(allocator: Allocator) !Self {
            return Self{ .allocator = allocator, .modules = try ArrayList(LinearLayer(T)).init(allocator) };
        }

        pub fn deinit(self: Self) void {
            for (self.modules.items) |layer| {
                layer.deinit();
            }
            self.modules.deinit();
        }

        pub fn forward(self: Self, A: Matrix(T)) !void {
            var B: Matrix(T) = Matrix(T).init(1, 1, self.allocator);
            for (self.modules.items) |mod| {
                const tmp: Matrix(T) = try mod.forward(A);
                defer tmp.deinit();
                // Copy the result to the master Matrix
                B = try Matrix(T).init(tmp.rows, tmp.cols, self.allocator);
                for (0..tmp.items.len) |i| {
                    B.items[i] = tmp.items[i];
                }
            }
        }
    };
}
