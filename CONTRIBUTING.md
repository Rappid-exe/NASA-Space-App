# Contributing to Exoplanet Classifier

Thank you for your interest in contributing! This project was created for the NASA Space Apps Challenge 2025.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NASA-Space-App.git
   cd NASA-Space-App
   ```

3. **Set up the environment**
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```

4. **Download NASA data** (see [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md))

5. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 🎯 Areas for Contribution

### High Priority
- [ ] Add comprehensive unit tests
- [ ] Improve error handling
- [ ] Add feature importance visualization
- [ ] Support for TESS and K2 datasets
- [ ] Docker containerization

### Medium Priority
- [ ] Hyperparameter tuning interface
- [ ] Model comparison dashboard
- [ ] Export predictions to various formats
- [ ] Add more example datasets
- [ ] Performance optimizations

### Low Priority
- [ ] Dark/light mode toggle
- [ ] Internationalization (i18n)
- [ ] Progressive Web App (PWA) features
- [ ] Real-time model retraining
- [ ] Cloud deployment guides

---

## 📝 Development Guidelines

### Code Style

**Python:**
- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and small

**TypeScript/React:**
- Use functional components
- Follow React hooks best practices
- Use TypeScript types (no `any`)
- Keep components under 200 lines

### Testing

Before submitting a PR:
```bash
# Test API
python -m pytest tests/

# Test frontend
cd frontend && npm test

# Manual testing
python -m uvicorn api.main:app --reload
cd frontend && npm run dev
```

### Commit Messages

Use conventional commits:
```
feat: add feature importance visualization
fix: resolve model loading issue
docs: update setup instructions
test: add unit tests for data processor
refactor: simplify inference service
```

---

## 🐛 Reporting Issues

When reporting issues, please include:
- Operating system
- Python version
- Node.js version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

---

## 💡 Suggesting Features

We welcome feature suggestions! Please:
1. Check existing issues first
2. Describe the use case
3. Explain the expected behavior
4. Consider implementation complexity

---

## 🔄 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG.md**
5. **Request review** from maintainers

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear

---

## 📚 Resources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## 🏆 Contributors

Thank you to all contributors! Your work helps advance exoplanet discovery.

---

## 📧 Contact

For questions or discussions:
- Open an issue on GitHub
- Join our discussions tab
- Email: [your-email@example.com]

---

**Happy coding! 🚀🪐**
