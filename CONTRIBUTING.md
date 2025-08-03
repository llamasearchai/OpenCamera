# Contributing to OpenCam Auto Exposure Algorithm

Thank you for your interest in contributing to OpenCam! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you agree to uphold these standards:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences
- Show empathy towards other community members

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- C++17 compatible compiler (GCC 8+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- Python 3.7+
- Git for version control
- Basic understanding of computer vision concepts

### Development Environment

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/OpenCamera.git
   cd OpenCamera
   ```

2. **Set up upstream remote**
   ```bash
   git remote add upstream https://github.com/llamasearchai/OpenCamera.git
   ```

3. **Install dependencies**
   ```bash
   # Install system dependencies (Ubuntu/Debian)
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libopencv-dev libspdlog-dev

   # Install Python dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Build the project**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON
   make -j$(nproc)
   ```

5. **Run tests**
   ```bash
   ctest --output-on-failure
   cd ../python
   pytest tests/
   ```

## Contribution Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Features**: Add new functionality
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Create usage examples
- **Benchmarks**: Add performance benchmarks

### Before You Start

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: For significant changes, create an issue first to discuss
3. **Assign yourself**: Comment on the issue to indicate you're working on it
4. **Stay updated**: Keep your fork synchronized with upstream

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. **Make your changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run C++ tests
   cd build && ctest
   
   # Run Python tests
   cd python && pytest
   
   # Run benchmarks
   cd build && ./auto_exposure_benchmark
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new metering algorithm
   
   - Implement center-weighted metering
   - Add configuration parameters
   - Include comprehensive tests
   - Update documentation
   
   Closes #123"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

## Coding Standards

### C++ Guidelines

- **Standard**: Use C++17 features appropriately
- **Style**: Follow Google C++ Style Guide with modifications:
  - Use 4 spaces for indentation
  - Maximum line length: 100 characters
  - Use `snake_case` for variables and functions
  - Use `PascalCase` for classes and structs
  - Use `UPPER_CASE` for constants and macros

- **Example**:
  ```cpp
  class AutoExposureController {
  public:
      explicit AutoExposureController(const Parameters& params);
      
      float compute_exposure(const cv::Mat& image);
      
  private:
      static constexpr float DEFAULT_TARGET_BRIGHTNESS = 0.5f;
      
      Parameters parameters_;
      std::unique_ptr<SceneAnalyzer> scene_analyzer_;
  };
  ```

### Python Guidelines

- **Standard**: Follow PEP 8
- **Type hints**: Use type annotations
- **Documentation**: Use Google-style docstrings
- **Formatting**: Use Black for code formatting
- **Imports**: Use isort for import organization

- **Example**:
  ```python
  def compute_exposure(
      self, 
      image: np.ndarray, 
      frame_number: int = 0
  ) -> Tuple[float, Dict[str, Any]]:
      """Compute optimal exposure for the given image.
      
      Args:
          image: Input image as numpy array
          frame_number: Frame sequence number
          
      Returns:
          Tuple of (exposure_value, scene_analysis_dict)
          
      Raises:
          ValueError: If image format is invalid
      """
      pass
  ```

### Documentation Standards

- **Comments**: Explain why, not what
- **API docs**: Document all public interfaces
- **Examples**: Include usage examples
- **Architecture**: Document design decisions

## Testing

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical paths
4. **Regression Tests**: Prevent performance degradation

### Writing Tests

#### C++ Tests (Google Test)
```cpp
TEST(AutoExposureTest, ComputesCorrectExposure) {
    AutoExposureController controller;
    cv::Mat test_image = create_test_image(640, 480);
    
    float exposure = controller.compute_exposure(test_image);
    
    EXPECT_GT(exposure, 0.0f);
    EXPECT_LT(exposure, 1.0f);
}
```

#### Python Tests (pytest)
```python
def test_compute_exposure_valid_input():
    controller = AutoExposure()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    exposure = controller.compute_exposure(image)
    
    assert 0.0 < exposure < 1.0
```

### Test Requirements

- **Coverage**: Maintain >90% code coverage
- **Performance**: Include benchmark tests for critical functions
- **Edge cases**: Test boundary conditions and error cases
- **Documentation**: Test examples in documentation

## Documentation

### Types of Documentation

1. **API Documentation**: Auto-generated from code comments
2. **User Guide**: How to use the library
3. **Developer Guide**: How to contribute and extend
4. **Examples**: Practical usage examples
5. **Architecture**: System design and decisions

### Documentation Tools

- **C++**: Doxygen for API documentation
- **Python**: Sphinx with autodoc
- **Markdown**: For guides and README files
- **OpenAPI**: For REST API documentation

## Pull Request Process

### Commit Message Policy

- Do not use emojis in commit messages.
- Keep subject lines concise (max 72 characters), in imperative mood, and without a trailing period.
- Prefer Conventional Commits style (feat:, fix:, docs:, refactor:, test:, build:, ci:, chore:) or a clear, descriptive subject.
- Reference issues when applicable using "Fixes #123".

Our CI enforces a no-emoji and professional commit policy. PRs or pushes that violate these rules may fail the commit policy workflow.

### Commit Message Policy

- Do not use emojis in commit messages.
- Keep subject lines concise (max 72 characters), in imperative mood, and without a trailing period.
- Prefer Conventional Commits style (feat:, fix:, docs:, refactor:, test:, build:, ci:, chore:) or a clear, descriptive subject.
- Reference issues when applicable using "Fixes #123".

Our CI enforces a no-emoji and professional commit policy. PRs or pushes that violate these rules may fail the commit policy workflow.


### PR Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] PR description explains changes
- [ ] No merge conflicts with main branch

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Other (please describe)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User guide updated (if applicable)

## Performance Impact
Describe any performance implications

## Breaking Changes
List any breaking changes

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs automatically
2. **Code review**: Maintainers review code quality and design
3. **Testing**: Comprehensive test suite validation
4. **Documentation**: Review of documentation updates
5. **Approval**: At least one maintainer approval required

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Environment**: OS, compiler, library versions
- **Steps to reproduce**: Detailed reproduction steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error output
- **Minimal example**: Code that reproduces the issue

### Feature Requests

For feature requests, include:

- **Use case**: Why is this feature needed
- **Proposed solution**: How should it work
- **Alternatives**: Other solutions considered
- **Implementation**: Thoughts on implementation

### Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `performance`: Performance-related issues
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: Direct contact with maintainers

### Recognition

We recognize contributors through:

- **Contributors file**: List of all contributors
- **Release notes**: Acknowledgment in releases
- **GitHub insights**: Automatic contribution tracking

### Mentorship

New contributors can:

- Start with `good first issue` labels
- Ask questions in discussions
- Request mentorship from maintainers
- Pair program on complex features

## Release Process (Maintainers)

- Version bump: Update versions in [`pyproject.toml`](pyproject.toml) and [`setup.py`](setup.py:52).
- Update [`CHANGELOG.md`](CHANGELOG.md) with the new version and date section.
- Create and merge a release PR from `main` with professional commit messages.
- Tag the release as `vX.Y.Z`. Pushing the tag triggers the release workflow which:
  - Builds and tests across C++ and Python,
  - Publishes to PyPI (requires `PYPI_API_TOKEN` in repository secrets),
  - Creates a GitHub Release with notes extracted from the changelog.
- Verify GitHub Release and PyPI publication.

## Release Process (Maintainers)

- Version bump: Update versions in [`pyproject.toml`](pyproject.toml) and [`setup.py`](setup.py:52).
- Update [`CHANGELOG.md`](CHANGELOG.md) with the new version and date section.
- Create and merge a release PR from `main` with professional commit messages.
- Tag the release as `vX.Y.Z`. Pushing the tag triggers the release workflow which:
  - Builds and tests across C++ and Python,
  - Publishes to PyPI (requires `PYPI_API_TOKEN` in repository secrets),
  - Creates a GitHub Release with notes extracted from the changelog.
- Verify GitHub Release and PyPI publication.

## Getting Help

If you need help:

1. Check existing documentation
2. Search existing issues
3. Ask in GitHub Discussions
4. Contact maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to OpenCam!** Your efforts help make this project better for everyone.

## Contact

**Nik Jois** - [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai)

Project Link: [https://github.com/llamasearchai/OpenCamera](https://github.com/llamasearchai/OpenCamera) 