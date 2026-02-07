# cblockdiff

cblockdiff — a maintained fork of blockdiff with continuous block diff support

Fast block-level file diffs (e.g. for VM disk images) using CoW filesystem metadata

## Usage

### File snapshots

> [!NOTE]
> this needs to be used within a XFS partition, not EXT4.

Creating a snapshot:

```
blockdiff create output.bdiff target.img --base base.img
```

Applying a snapshot:

```
blockdiff apply input.bdiff target.img --base base.img
```

### Multiple sequential diffs

You can chain multiple diffs together using comma-separated base files:

Creating a diff from a base + previous diffs:

```
blockdiff create diff3.bdiff target.img --base base.img,diff1.bdiff,diff2.bdiff
```

Applying multiple diffs sequentially:

```
blockdiff apply diff3.bdiff target.img --base base.img,diff1.bdiff,diff2.bdiff
```

This allows incremental snapshots where each diff builds on previous ones.

### Compactifying sparse files

You can also use blockdiff without a base image to "compactify" sparse files for storage. A sparse file might have a size of 100GB but only 10GB of data. The blockdiff tool creates a compact 10GB file containing only the actual data.

```
blockdiff create compact.bdiff target.img
blockdiff apply compact.bdiff target.img
```
