# Project Structure

Clean, organized structure for the Exoplanet Classifier project.

## 📂 Directory Layout

```
NASA-Space-App/
│
├── 📁 api/                    # Backend API (FastAPI)
│   ├── main.py               # API endpoints
│   ├── inference_service.py  # Classification service
│   └── README.md             # API documentation
│
├── 📁 data/                   # Data pipeline
│   ├── dataset_downloader.py # Download NASA data
│   ├── dataset_loader.py     # Load datasets
│   ├── dataset_validator.py  # Validate data
│   └── data_processor.py     # Process & engineer features
│
├── 📁 models/                 # Machine Learning
│   ├── base_classifier.py    # Base ML interface
│   ├── random_forest_classifier.py
│   ├── neural_network_classifier.py
│   ├── svm_classifier.py
│   ├── model_trainer.py      # Training pipeline
│   ├── model_evaluator.py    # Evaluation metrics
│   └── model_registry.py     # Model management
│
├── 📁 frontend/               # Web Interface (Next.js)
│   ├── app/                  # Pages (App Router)
│   │   ├── page.tsx         # Homepage
│   │   ├── classify/        # Classification page
│   │   └── upload/          # Batch upload page
│   ├── components/          # React components
│   │   ├── Dashboard.tsx
│   │   ├── ClassificationForm.tsx
│   │   ├── ResultsDisplay.tsx
│   │   └── FileUpload.tsx
│   ├── lib/                 # Utilities
│   │   ├── api.ts          # API client
│   │   └── types.ts        # TypeScript types
│   └── README.md           # Frontend docs
│
├── 📁 tests/                 # Test files
│   ├── test_data_ingestion.py
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   ├── test_model_registry.py
│   └── test_api_structure.py
│
├── 📁 examples/              # Example scripts
│   └── example_model_registry_usage.py
│
├── 📁 docs/                  # Documentation
│   ├── README.md            # Docs index
│   ├── API_QUICKSTART.md
│   ├── FRONTEND_SETUP_GUIDE.md
│   └── [other docs]
│
├── 📁 .kiro/                 # Kiro specs
│   └── specs/
│       └── exoplanet-identifier/
│
├── 📄 README.md              # Main project README
├── 📄 SETUP.md               # Setup instructions
├── 📄 requirements.txt       # Python dependencies
└── 📄 HackathonContext.md    # Hackathon info
```

## 🎯 Key Files

### Root Level
- **README.md** - Project overview and quick start
- **SETUP.md** - Detailed setup instructions
- **requirements.txt** - Python dependencies
- **HackathonContext.md** - Hackathon context

### API
- **api/main.py** - FastAPI application with endpoints
- **api/inference_service.py** - Classification logic
- **api/README.md** - API documentation

### Frontend
- **frontend/package.json** - Node.js dependencies
- **frontend/app/page.tsx** - Homepage
- **frontend/README.md** - Frontend documentation

### Models
- **models/model_registry.py** - Model versioning & storage
- **models/model_trainer.py** - Training pipeline
- **models/*_classifier.py** - ML model implementations

### Data
- **data/data_processor.py** - Complete data pipeline
- **data/dataset_*.py** - Data ingestion utilities

## 🗂️ Organization Principles

1. **Separation of Concerns** - Each directory has a clear purpose
2. **Documentation Co-location** - READMEs in relevant directories
3. **Test Isolation** - All tests in `tests/` directory
4. **Example Isolation** - Example scripts in `examples/`
5. **Doc Centralization** - All docs in `docs/` directory

## 🚀 Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Start the API | `api/` |
| Start the frontend | `frontend/` |
| Train a model | `models/` |
| Process data | `data/` |
| Run tests | `tests/` |
| Read docs | `docs/` |
| See examples | `examples/` |

## 📝 File Naming Conventions

- **Python files**: `snake_case.py`
- **TypeScript files**: `PascalCase.tsx` (components), `camelCase.ts` (utilities)
- **Documentation**: `UPPERCASE.md` (root), `Title_Case.md` (docs/)
- **Tests**: `test_*.py`
- **Examples**: `example_*.py`

## 🧹 Keeping It Clean

### What Goes Where

**Root Directory** (minimal)
- Main README
- Setup guide
- Requirements
- License
- .gitignore

**docs/** (all documentation)
- Setup guides
- Implementation summaries
- Quick references
- Technical docs

**tests/** (all test files)
- Unit tests
- Integration tests
- API tests

**examples/** (example scripts)
- Usage examples
- Demo scripts

### What NOT to Put in Root

- ❌ Implementation summaries
- ❌ Component-specific guides
- ❌ Test files
- ❌ Example scripts
- ❌ Temporary files

## 🔄 Maintenance

When adding new files:
1. Determine the correct directory
2. Follow naming conventions
3. Update relevant README
4. Keep root directory clean

## 📚 Related Documentation

- [Main README](../README.md)
- [Setup Guide](../SETUP.md)
- [Documentation Index](../docs/README.md)
