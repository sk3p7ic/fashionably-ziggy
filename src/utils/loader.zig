const std = @import("std");
const matrices = @import("./matrices.zig");
const Matrix = matrices.Matrix;

const FILE_PATH = "dataset/fashion-mnist.csv";

/// Stores a raw datapoint.
pub const Datapoint = struct { value: [784]u8, label: u8 };

/// Stores the dataset, separated by items and their labels.
pub const Dataset = struct { items: Matrix(f32), labels: Matrix(f32) };

/// Reads the input dataset and converts it into a dataset.
pub fn read_train_file(allocator: std.mem.Allocator) !Dataset {
    const n_datapoints = 60000; // The number of datapoints in this dataset
    // Open and read the input file
    const file = try std.fs.cwd().openFile(FILE_PATH, .{});
    defer file.close();
    const buff_size = 133128873 * @sizeOf(u8);
    const file_buff = try file.readToEndAlloc(allocator, buff_size);
    defer allocator.free(file_buff);
    // Get the lines in the file
    var row_iter = std.mem.splitAny(u8, file_buff, "\n");
    var data: []Datapoint = try allocator.alloc(Datapoint, n_datapoints);
    defer allocator.free(data);
    // Parse each row of the file and convert to Datapoints
    var data_idx: usize = 0;
    while (row_iter.next()) |row| : (data_idx += 1) {
        var tokens = std.mem.tokenizeAny(u8, row, ",");
        if (tokens.next()) |label| {
            var idx: usize = 0;
            var values: [784]u8 = [_]u8{0} ** 784;
            while (tokens.next()) |tok| : (idx += 1) {
                const stripped_token = std.mem.trim(u8, tok, "\n\r");
                values[idx] = try std.fmt.parseInt(u8, stripped_token, 10);
            }
            data[data_idx] = Datapoint{ .label = try std.fmt.parseInt(u8, label, 10), .value = values };
        }
    }
    // Shuffle the data
    var rng = std.rand.DefaultPrng.init(0);
    std.rand.Random.shuffle(rng.random(), Datapoint, data);
    // Convert the list of Datapoints into a Dataset
    const dataset_items = try Matrix(f32).init(n_datapoints, 784, allocator);
    const dataset_labels = try Matrix(f32).init(n_datapoints, 1, allocator);
    var dataset = Dataset{ .items = dataset_items, .labels = dataset_labels };
    for (0..data.len) |i| {
        // Convert attributes from u8 to f32
        var values: [784]f32 = [_]f32{0.0} ** 784;
        for (0..784) |vi| {
            values[vi] = @as(f32, @floatFromInt(data[i].value[vi]));
        }
        // Add to matrix
        try dataset.items.insertRow(&values, i);
        (try dataset.labels.at(i, 0)).* = @as(f32, @floatFromInt(data[i].label));
    }
    return dataset;
}

test "Can read and parse training file" {
    const allocator = std.testing.allocator;
    const data = try read_train_file(allocator);
    defer data.items.deinit();
    defer data.labels.deinit();
}
