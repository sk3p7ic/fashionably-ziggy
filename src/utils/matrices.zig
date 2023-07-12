const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MatrixError = error{
    IndexOutOfBounds,
    ShapeMismatch,
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

        pub fn dot(self: Self, mtx_b: Matrix(T)) !Self {
            if (self.cols != mtx_b.rows) {
                return MatrixError.ShapeMismatch;
            }
            const prod = try Matrix(T).init(self.rows, mtx_b.cols, self.allocator);
            prod.fill(0);
            for (0..self.rows) |i| {
                for (0..self.cols) |k| {
                    for (0..mtx_b.cols) |j| {
                        (try prod.at(i, j)).* += (try self.at(i, k)).* * (try mtx_b.at(k, j)).*;
                    }
                }
            }
            return prod;
        }

        pub fn sum(self: Self, mtx_b: Matrix(T)) !Self {
            if (self.rows != mtx_b.rows or self.cols != mtx_b.cols) {
                return MatrixError.ShapeMismatch;
            }
            const mtx = try Matrix(T).init(self.rows, self.cols, self.allocator);
            for (self.items, 0..) |e, i| {
                mtx.items[i] = e + mtx_b.items[i];
            }
            return mtx;
        }

        pub fn sub(self: Self, mtx_b: Matrix(T)) !Self {
            if (self.rows != mtx_b.rows or self.cols != mtx_b.cols) {
                return MatrixError.ShapeMismatch;
            }
            const mtx = try Matrix(T).init(self.rows, self.cols, self.allocator);
            for (self.items, 0..) |e, i| {
                mtx.items[i] = e - mtx_b.items[i];
            }
            return mtx;
        }

        pub fn equivCheck(self: Self, mtx_b: Matrix(T)) bool {
            // Check shape
            if (self.rows != mtx_b.rows or self.cols != mtx_b.cols) {
                return false;
            }
            // Check elements
            for (self.items, 0..) |e, i| {
                if (e != mtx_b.items[i]) {
                    return false;
                }
            }
            return true;
        }

        pub fn subdivide(self: Self, rows: usize, cols: usize, row_start: usize, col_start: usize) !Self {
            if (rows > self.rows or cols > self.cols) {
                return MatrixError.ShapeMismatch;
            }
            if (row_start >= self.rows or col_start >= self.cols) {
                return MatrixError.IndexOutOfBounds;
            }
            const mtx = try Matrix(T).init(rows, cols, self.allocator);
            for (row_start..rows) |r| {
                for (col_start..cols) |c| {
                    (try mtx.at(r, c)).* = (try self.at(r, c)).*;
                }
            }
            return mtx;
        }
    };
}

pub fn IdentityMatrix(comptime T: type, n: usize, allocator: Allocator) !Matrix(T) {
    const ident = try Matrix(T).init(n, n, allocator);
    ident.fill(0);
    for (0..n) |r| {
        for (0..n) |c| {
            if (r == c) {
                (try ident.at(r, c)).* = 1;
            }
        }
    }
    return ident;
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

test "Can perform dot product of two matrices." {
    const allocator = std.testing.allocator;
    const mtx1 = try Matrix(u8).init(2, 3, allocator);
    defer mtx1.deinit();
    const mtx2 = try Matrix(u8).init(3, 1, allocator);
    defer mtx2.deinit();
    try mtx1.insertRowConst(&[3]u8{ 1, 2, 3 }, 0);
    try mtx1.insertRowConst(&[3]u8{ 4, 5, 6 }, 1);
    (try mtx2.at(0, 0)).* = 7;
    (try mtx2.at(1, 0)).* = 8;
    (try mtx2.at(2, 0)).* = 9;
    // Test if dot product works for happy case
    const mtx3 = try mtx1.dot(mtx2);
    defer mtx3.deinit();
    try std.testing.expect((try mtx3.at(0, 0)).* == 50);
    try std.testing.expect((try mtx3.at(1, 0)).* == 122);
    // Test if dot product can detect shape mismatches
    const mtx4 = mtx1.dot(mtx3);
    try std.testing.expectError(MatrixError.ShapeMismatch, mtx4);
}

test "Equivalency check for matrices" {
    const allocator = std.testing.allocator;
    const mtx1 = try Matrix(u16).init(3, 3, allocator);
    defer mtx1.deinit();
    const mtx2 = try Matrix(u16).init(3, 3, allocator);
    defer mtx2.deinit();
    mtx1.fill(69);
    mtx2.fill(69);
    try std.testing.expect(mtx1.equivCheck(mtx2));
    mtx2.items[7] = 420; // Change an element
    try std.testing.expect(!mtx2.equivCheck(mtx1));
}

test "Matrix addition and subtraction works" {
    const allocator = std.testing.allocator;
    const mtx1 = try Matrix(u8).init(3, 3, allocator);
    defer mtx1.deinit();
    const mtx2 = try Matrix(u8).init(3, 3, allocator);
    defer mtx2.deinit();
    const sum_mtx = try Matrix(u8).init(3, 3, allocator);
    defer sum_mtx.deinit();
    mtx1.fill(1);
    mtx2.fill(2);
    sum_mtx.fill(3);
    const sum_res = try mtx1.sum(mtx2);
    defer sum_res.deinit();
    const sub_res = try sum_mtx.sub(mtx2);
    defer sub_res.deinit();
    try std.testing.expect(sum_mtx.equivCheck(sum_res));
    try std.testing.expect(mtx1.equivCheck(sub_res));
}

test "Can make Identity Matrix" {
    const allocator = std.testing.allocator;
    const ident = try IdentityMatrix(u8, 3, allocator);
    defer ident.deinit();
    const comparison_ident = try Matrix(u8).init(3, 3, allocator);
    defer comparison_ident.deinit();
    comparison_ident.fill(0);
    comparison_ident.items[0] = 1;
    comparison_ident.items[4] = 1;
    comparison_ident.items[8] = 1;
    try std.testing.expect(ident.equivCheck(comparison_ident));
}

test "Can subdivide matrices" {
    const allocator = std.testing.allocator;
    const mtx1 = try Matrix(u8).init(4, 4, allocator);
    defer mtx1.deinit();
    try mtx1.insertRowConst(&[_]u8{ 2, 2, 4, 4 }, 0);
    try mtx1.insertRowConst(&[_]u8{ 2, 2, 4, 4 }, 1);
    try mtx1.insertRowConst(&[_]u8{ 4, 4, 4, 4 }, 2);
    try mtx1.insertRowConst(&[_]u8{ 4, 4, 4, 4 }, 3);
    const comp_mtx = try Matrix(u8).init(2, 2, allocator);
    defer comp_mtx.deinit();
    try comp_mtx.insertRowConst(&[_]u8{ 2, 2 }, 0);
    try comp_mtx.insertRowConst(&[_]u8{ 2, 2 }, 1);
    const mtx2 = try mtx1.subdivide(2, 2, 0, 0);
    defer mtx2.deinit();
    try std.testing.expect(comp_mtx.equivCheck(mtx2));
}
