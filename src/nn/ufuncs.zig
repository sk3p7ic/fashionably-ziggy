const std = @import("std");

pub fn relu(comptime T: type, x: T) T {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}
