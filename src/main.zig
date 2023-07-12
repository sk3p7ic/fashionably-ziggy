const std = @import("std");
const loader = @import("./utils/loader.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var bw = std.io.bufferedWriter(std.io.getStdOut().writer());
    const stdout = bw.writer();
    try stdout.print("Loading input data...\n", .{});
    try bw.flush();
    var timer = try std.time.Timer.start();

    const data = try loader.read_train_file(allocator);
    // TODO: Maybe deinit after the subdivisions? Do we still need this data?
    defer data.items.deinit();
    defer data.labels.deinit();
    {
        const time_taken = timer.read() / std.time.ns_per_s;
        const data_attrs = try data.items.sizeAsStr();
        const data_labels = try data.labels.sizeAsStr();
        try stdout.print("Loaded {s} attributes with {s} labels in {d} seconds.\n", .{ data_attrs, data_labels, time_taken });
        try bw.flush();
    }
    try stdout.print("Subdividing datasets.\n", .{});
    try bw.flush();
    const train_items = try data.items.subdivide(10000, data.items.cols, 0, 0);
    defer train_items.deinit();
    const train_labels = try data.labels.subdivide(10000, 1, 0, 0);
    defer train_labels.deinit();
    const test_items = try data.items.subdivide(50000, data.items.cols, 10000, 0);
    defer test_items.deinit();
    const test_labels = try data.labels.subdivide(50000, 1, 10000, 0);
    defer test_labels.deinit();
    try stdout.print("Divided:\n  Train Items: {s}\n  Train Labels: {s}\n  Test Items: {s}\n  Tests Labels: {s}\n", .{ try train_items.sizeAsStr(), try train_labels.sizeAsStr(), try test_items.sizeAsStr(), try test_labels.sizeAsStr() });
    try bw.flush();
}
