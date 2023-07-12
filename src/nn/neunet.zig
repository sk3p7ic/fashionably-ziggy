const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

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
        modules: ArrayList(T),

        pub fn init(allocator: Allocator) !Self {
            return Self{ .allocator = allocator, .modules = try ArrayList(T).init(allocator) };
        }

        pub fn deinit(self: Self) void {
            self.modules.deinit();
        }
    };
}
