const std = @import("std");
const loader = @import("./utils/loader.zig");
const neunet = @import("./nn/neunet.zig");
const ufuncs = @import("./nn/ufuncs.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var bw = std.io.bufferedWriter(std.io.getStdOut().writer());
    const stdout = bw.writer();
    try stdout.print("Loading input data...\n", .{});
    try bw.flush();
    var timer = try std.time.Timer.start();

    const data = try loader.read_train_file(allocator);
    defer data.items.deinit();
    defer data.labels.deinit();
    {
        const time_taken = timer.lap() / std.time.ns_per_s;
        const data_attrs = try data.items.sizeAsStr();
        const data_labels = try data.labels.sizeAsStr();
        try stdout.print("Loaded {s} attributes with {s} labels in {d} seconds.\n", .{ data_attrs, data_labels, time_taken });
        try bw.flush();
    }
    try stdout.print("Subdividing datasets.\n", .{});
    try bw.flush();
    const train_items = try (try data.items.subdivide(10000, data.items.cols, 0, 0)).transpose();
    defer train_items.deinit();
    const train_labels = try (try data.labels.subdivide(10000, 1, 0, 0)).transpose();
    defer train_labels.deinit();
    const test_items = try (try data.items.subdivide(50000, data.items.cols, 10000, 0)).transpose();
    defer test_items.deinit();
    const test_labels = try (try data.labels.subdivide(50000, 1, 10000, 0)).transpose();
    defer test_labels.deinit();
    {
        const time_taken = timer.read() / std.time.ns_per_s;
        const trnis = try train_items.sizeAsStr();
        const trnls = try train_labels.sizeAsStr();
        const tstis = try test_items.sizeAsStr();
        const tstls = try test_labels.sizeAsStr();
        try stdout.print("Divided (Took {d} seconds):\n  Train Items: {s}\n  Train Labels: {s}\n  Test Items: {s}\n  Tests Labels: {s}\n", .{ time_taken, trnis, trnls, tstis, tstls });
        try bw.flush();
    }
    const nn = neunet.NN(f32).init([2]ufuncs.ActivationFunctionTag{ .relu, .softmax }, allocator);
    _ = nn;
}
