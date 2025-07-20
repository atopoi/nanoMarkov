# Extra Tools and Legacy Utilities

This directory contains legacy tools and utilities that are not part of the core nanoMarkov workflow.

## Makefile (Deprecated)

The Makefile in this directory was the original build automation tool for nanoMarkov. It is now deprecated in favor of the simpler `train_mm.sh` script.

### Known Issues with the Makefile

- Does not handle `wandb_log` configuration properly (defaults to True in configs)
- Contains hardcoded seeds in some targets
- Complex dependency management that can be confusing
- Does not follow the two-step process (data generation then training) clearly

### Why We Moved Away

The Makefile became overly complex as the project grew. The new approach using `train_mm.sh` provides:
- Clear separation between data generation and training
- Better control over wandb logging (disabled by default)
- Simpler command-line interface
- Easier to understand and modify

### Using the Legacy Makefile

If you still want to use the Makefile, you can run it from the project root:

```bash
cd ..  # Go to project root
make -f extra/Makefile help
```

However, we strongly recommend using the modern workflow described in the [TRAINING_GUIDE.md](../TRAINING_GUIDE.md).

### Contributing

If you'd like to fix the Makefile and bring it up to date with the current best practices, contributions are welcome! The main improvements needed are:

1. Handle `wandb_log` configuration properly
2. Remove hardcoded seeds
3. Simplify the dependency management
4. Better align with the two-step training process

Please submit a pull request if you make improvements to the Makefile.