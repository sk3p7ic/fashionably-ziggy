const std = @import("std");
const nn_layers = @import("layers.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const LinearLayer = nn_layers.LinearLayer;

pub fn relu(comptime T: type, x: T) T {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

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
    };
}
