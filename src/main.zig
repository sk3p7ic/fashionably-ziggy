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
    defer data.items.deinit();
    defer data.labels.deinit();
    {
        const time_taken = timer.read() / std.time.ns_per_s;
        const data_attrs = try data.items.sizeAsStr();
        const data_labels = try data.labels.sizeAsStr();
        try stdout.print("Loaded {s} attributes with {s} labels in {d} seconds.\n", .{ data_attrs, data_labels, time_taken });
        try bw.flush();
    }
}
