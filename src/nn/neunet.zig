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

        pub fn deinit(self: Self) void {
            self.A1.deinit();
            self.A2.deinit();
            self.Z1.deinit();
            self.Z2.deinit();
        }
    };
}

pub fn NN(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        layer1: LinearLayer(T, nn_funcs.relu_activation),
        layer2: LinearLayer(T, nn_funcs.softmax_activation),

        pub fn init(allocator: Allocator) !Self {
            return Self{ .allocator = allocator, .layer1 = try LinearLayer(T, nn_funcs.relu_activation).init(86, 784, allocator), .layer2 = try LinearLayer(T, nn_funcs.softmax_activation).init(10, 86, allocator) };
        }

        pub fn deinit(self: Self) void {
            self.layer1.deinit();
            self.layer2.deinit();
        }

        fn forward(self: Self, A: *const Matrix(T)) !NNForwardResult(T) {
            const layer1 = try self.layer1.forward(A);
            defer layer1.deinit();
            const layer2 = try self.layer2.forward(&layer1.A);
            defer layer2.deinit();
            // Copy the results to a memory-safe struct
            const epoch_result = try NNForwardResult(T).init(layer1.Z, layer1.A, layer2.Z, layer2.A);
            return epoch_result;
        }

        fn backprop(self: Self, forward_res: *const NNForwardResult(T), A: *const Matrix(T), actuals: *const Matrix(T), lr: T) !void {
            const m: T = @as(T, @floatFromInt(forward_res.A2.cols));
            const one_hot = try Matrix(T).init(forward_res.A2.rows, actuals.cols, self.allocator);
            defer one_hot.deinit();
            one_hot.fill(0);
            for (actuals.items, 0..) |e, i| {
                (try one_hot.at(@as(usize, @intFromFloat(e)), i)).* = 1;
            }
            const dz2 = try forward_res.A2.sub(one_hot);
            defer dz2.deinit();
            const dw2 = (try dz2.dot(try forward_res.A1.transpose()));
            defer dw2.deinit();
            for (dw2.items, 0..) |e, i| {
                dw2.items[i] = e / m;
            }
            const db2 = try Matrix(T).init(forward_res.A2.rows, 1, self.allocator);
            defer db2.deinit();
            db2.fill(0);
            for (0..dz2.rows) |r| {
                for (0..dz2.cols) |c| {
                    (try db2.at(r, 0)).* += (try dz2.at(r, c)).*;
                }
            }
            for (db2.items, 0..) |e, i| {
                db2.items[i] = e / m;
            }
            const dz1 = try (try self.layer2.weights.transpose()).dot(dz2);
            defer dz1.deinit();
            const dw1 = try dz1.dot(try A.transpose());
            defer dw1.deinit();
            for (dw1.items, 0..) |e, i| {
                dw1.items[i] = e / m;
            }
            const db1 = try Matrix(T).init(dz1.rows, 1, self.allocator);
            defer db1.deinit();
            db1.fill(0);
            for (0..dz1.rows) |r| {
                for (0..dz1.cols) |c| {
                    (try db1.at(r, 0)).* += (try dz1.at(r, c)).*;
                }
            }
            for (db1.items, 0..) |e, i| {
                db1.items[i] = e / m;
            }
            // Update values in the layers
            for ((try self.layer1.weights.sub(try dw1.scalarMult(lr))).items, 0..) |e, i| {
                self.layer1.weights.items[i] = e;
            }
            for ((try self.layer1.biases.sub(try db1.scalarMult(lr))).items, 0..) |e, i| {
                self.layer1.biases.items[i] = e;
            }
            for ((try self.layer2.weights.sub(try dw2.scalarMult(lr))).items, 0..) |e, i| {
                self.layer2.weights.items[i] = e;
            }
            for ((try self.layer2.biases.sub(try db2.scalarMult(lr))).items, 0..) |e, i| {
                self.layer2.biases.items[i] = e;
            }
        }

        fn accuracy_score(self: Self, preds: *const Matrix(T), actls: *const Matrix(T)) !f32 {
            const maxes = try Matrix(T).init(1, preds.cols, self.allocator);
            defer maxes.deinit();
            for (0..preds.cols) |c| {
                var max: T = 0;
                for (0..preds.rows) |r| {
                    const val = (try preds.at(r, c)).*;
                    if (val > max) {
                        max = val;
                    }
                }
                (try maxes.at(0, c)).* = max;
            }
            var sims: usize = 0;
            for (0..actls.rows) |r| {
                for (0..actls.cols) |c| {
                    const aptr = try actls.at(r, c);
                    const pptr = try preds.at(r, c);
                    if (aptr.* == pptr.*) {
                        sims += 1;
                    }
                }
            }
            return @as(f32, @floatFromInt(sims)) / @as(f32, @floatFromInt(actls.size()));
        }

        pub fn train(self: Self, data: *const Matrix(T), labels: *const Matrix(T), epochs: usize, lr: T) ![]f32 {
            var acc_scores = std.ArrayList(f32).init(self.allocator);
            for (0..epochs) |_| {
                const fw_res = try self.forward(data);
                defer fw_res.deinit();
                try self.backprop(&fw_res, data, labels, lr);
                try acc_scores.append(try self.accuracy_score(&fw_res.A2, labels));
            }
            return acc_scores.toOwnedSlice();
        }
    };
}
