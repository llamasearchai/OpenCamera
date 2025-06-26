## Description

**Summary**
Provide a brief summary of the changes in this PR.

**Motivation and Context**
Why is this change required? What problem does it solve?
If it fixes an open issue, please link to the issue here: Fixes #(issue)

## Type of Change

Please check the type of change your PR introduces:

- [ ] **Bug fix** (non-breaking change which fixes an issue)
- [ ] **New feature** (non-breaking change which adds functionality)
- [ ] **Breaking change** (fix or feature that would cause existing functionality to change)
- [ ] **Performance improvement** (code change that improves performance)
- [ ] **Refactoring** (code change that neither fixes a bug nor adds a feature)
- [ ] **Documentation** (changes to documentation only)
- [ ] **Tests** (adding missing tests or correcting existing tests)
- [ ] **Build/CI** (changes to build process or continuous integration)
- [ ] **Other** (please describe):

## Changes Made

**Core Changes:**
- [ ] Modified C++ core library
- [ ] Updated Python bindings
- [ ] Changed FastAPI endpoints
- [ ] Modified ML components
- [ ] Updated build system
- [ ] Changed CI/CD pipeline

**Detailed Changes:**
- List specific changes made
- Include file modifications
- Mention new dependencies
- Note configuration changes

## Testing

**Test Coverage:**
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing completed
- [ ] All existing tests pass

**Test Details:**
```bash
# Commands used for testing
make test
pytest python/tests/
./build/auto_exposure_benchmark
```

**Test Results:**
- Test coverage: [percentage]%
- Performance impact: [describe any changes]
- Memory usage: [if applicable]

## Performance Impact

**Benchmarks:**
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance degraded (justified)
- [ ] Performance not measured

**Benchmark Results:**
```
Include relevant benchmark results if applicable
```

## Breaking Changes

**API Changes:**
- [ ] No breaking changes
- [ ] Breaking changes documented below

**Breaking Change Details:**
If this PR introduces breaking changes, please describe:
- What APIs changed
- How to migrate existing code
- Deprecation timeline (if applicable)

## Documentation

**Documentation Updates:**
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] User guide updated
- [ ] No documentation needed

**Documentation Changes:**
Describe any documentation changes made.

## Dependencies

**New Dependencies:**
- [ ] No new dependencies
- [ ] New dependencies added (list below)

**Dependency Changes:**
- List any new dependencies
- Explain why they are needed
- Verify license compatibility

## Deployment

**Deployment Considerations:**
- [ ] No special deployment requirements
- [ ] Requires database migration
- [ ] Requires configuration changes
- [ ] Requires environment updates

**Deployment Notes:**
Any special instructions for deployment.

## Screenshots/Recordings

**Visual Changes:**
If applicable, add screenshots or recordings to demonstrate the changes.

**Before/After:**
Show the state before and after your changes.

## Checklist

**Code Quality:**
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

**Security:**
- [ ] I have considered security implications of my changes
- [ ] I have not introduced any security vulnerabilities
- [ ] I have validated all user inputs
- [ ] I have not exposed sensitive information

**Performance:**
- [ ] I have considered performance implications
- [ ] I have not introduced performance regressions
- [ ] I have optimized critical paths where possible
- [ ] I have measured performance impact where applicable

## Additional Notes

**Future Work:**
Any follow-up work that should be done after this PR.

**Known Issues:**
Any known issues or limitations with this implementation.

**References:**
- Link to relevant issues
- Link to design documents
- Link to external resources
- Link to related PRs

## Review Checklist for Maintainers

**Code Review:**
- [ ] Code follows project standards
- [ ] Logic is sound and efficient
- [ ] Error handling is appropriate
- [ ] Memory management is correct (for C++)
- [ ] Thread safety considered where applicable

**Testing:**
- [ ] Test coverage is adequate
- [ ] Tests are meaningful and comprehensive
- [ ] Performance tests included where needed
- [ ] Edge cases are covered

**Documentation:**
- [ ] Code is well-documented
- [ ] API changes are documented
- [ ] User-facing changes are documented
- [ ] Examples are provided where helpful

**Integration:**
- [ ] Changes integrate well with existing code
- [ ] No unnecessary dependencies introduced
- [ ] Backward compatibility maintained (or breaking changes justified)
- [ ] CI/CD pipeline passes

---

**Thank you for contributing to OpenCam! Your pull request helps make the project better for everyone.** 