const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MatrixError = error{
    IndexOutOfBounds,
};

pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        rows: usize,
        cols: usize,
        items: []T,
        allocator: Allocator,

        pub fn init(rows: usize, cols: usize, allocator: Allocator) !Self {
            return Self{ .rows = rows, .cols = cols, .items = try allocator.alloc(T, rows * cols), .allocator = allocator };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.items);
        }

        pub fn size(self: Self) usize {
            return self.rows * self.cols;
        }

        pub fn sizeAsStr(self: Self) ![]u8 {
            return try std.fmt.allocPrint(self.allocator, "[{d}x{d}] = {d}", .{ self.rows, self.cols, self.size() });
        }

        pub fn at(self: Self, row: usize, col: usize) MatrixError!*T {
            if (row >= self.rows or col >= self.cols) {
                return MatrixError.IndexOutOfBounds;
            }
            return &self.items[row * self.cols + col];
        }

        pub fn fill(self: Self, value: T) void {
            for (0..self.items.len) |i| {
                self.items[i] = value;
            }
        }

        pub fn insertRow(self: Self, row: []T, index: usize) MatrixError!void {
            if (index >= self.rows) {
                return MatrixError.IndexOutOfBounds;
            }
            for (0..row.len) |c| {
                const ptr = try self.at(index, c);
                ptr.* = row[c];
            }
        }
    };
}

test "Can alloc / dealloc Matrix" {
    const allocator = std.testing.allocator;
    const mtx = try Matrix(f32).init(2, 2, allocator);
    defer mtx.deinit();
}

test "Can insert row into Matrix" {
    const allocator = std.testing.allocator;
    const mtx = try Matrix(u8).init(3, 4, allocator);
    defer mtx.deinit();
    mtx.fill(0);
    var items = [_]u8{ 1, 2, 3, 4 };
    var all_test_items = [_]u8{ 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0 };
    try mtx.insertRow(&items, 1);
    const mtx_items = mtx.items;
    for (0..all_test_items.len) |i| {
        try std.testing.expect(all_test_items[i] == mtx_items[i]);
    }
}
