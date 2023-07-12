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

        pub fn insertRowConst(self: Self, row: []const T, index: usize) MatrixError!void {
            if (index >= self.rows) {
                return MatrixError.IndexOutOfBounds;
            }
            for (0..row.len) |c| {
                const ptr = try self.at(index, c);
                ptr.* = row[c];
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

        pub fn transpose(self: Self) !Self {
            var mtx = try Matrix(T).init(self.cols, self.rows, self.allocator);
            for (0..self.rows) |c| {
                for (0..self.cols) |r| {
                    (try mtx.at(r, c)).* = (try self.at(c, r)).*;
                }
            }
            return mtx;
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
    const items = [_]u8{ 1, 2, 3, 4 };
    const all_test_items = [_]u8{ 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0 };
    try mtx.insertRowConst(&items, 1);
    const mtx_items = mtx.items;
    for (0..all_test_items.len) |i| {
        try std.testing.expect(all_test_items[i] == mtx_items[i]);
    }
}

test "Can transpose a Matrix" {
    const allocator = std.testing.allocator;
    const mtx1 = try Matrix(u8).init(2, 3, allocator);
    defer mtx1.deinit();
    const r1 = [_]u8{ 1, 2, 3 };
    const r2 = [_]u8{ 4, 5, 6 };
    try mtx1.insertRowConst(&r1, 0);
    try mtx1.insertRowConst(&r2, 1);
    const mtx2 = try mtx1.transpose();
    defer mtx2.deinit();
    const expected = [_]u8{ 1, 4, 2, 5, 3, 6 };
    for (0..expected.len) |i| {
        try std.testing.expect(expected[i] == mtx2.items[i]);
    }
}
