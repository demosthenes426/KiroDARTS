"""
Comprehensive test runner for DARTS Stock Forecasting System.

This script orchestrates all comprehensive tests including integration tests,
performance benchmarks, synthetic data testing, and memory profiling.

Requirements: 5.3, 5.4
"""

import unittest
import sys
import os
import time
import warnings
import json
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
try:
    from TESTING_comprehensive_test_suite import TestComprehensiveIntegration
    from TESTING_performance_benchmarks import TestPerformanceBenchmarks
    from TESTING_synthetic_data_generators import TestSyntheticDataGenerators
    from TESTING_memory_profiling import TestMemoryProfiling, TestLoadTesting
    COMPREHENSIVE_TESTS_AVAILABLE = True
except ImportError as e:
    COMPREHENSIVE_TESTS_AVAILABLE = False
    print(f"Warning: Comprehensive test modules not available: {e}")

# Check for dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False


class ComprehensiveTestRunner:
    """Orchestrates comprehensive testing with reporting and analysis."""
    
    def __init__(self):
        """Initialize the comprehensive test runner."""
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.report_file = None
    
    def run_test_suite(
        self,
        test_categories: List[str] = None,
        output_file: str = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive test suite with specified categories.
        
        Args:
            test_categories: List of test categories to run
            output_file: File to save detailed results
            verbose: Whether to print verbose output
        
        Returns:
            Dictionary with test results and statistics
        """
        if test_categories is None:
            test_categories = ['integration', 'performance', 'synthetic', 'memory', 'load']
        
        self.start_time = time.time()
        self.report_file = output_file
        
        print("=" * 80)
        print("COMPREHENSIVE TEST SUITE FOR DARTS STOCK FORECASTING SYSTEM")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test categories: {', '.join(test_categories)}")
        print(f"Dependencies available: DARTS={DARTS_AVAILABLE}, psutil={PSUTIL_AVAILABLE}")
        print("=" * 80)
        
        # Run test categories
        for category in test_categories:
            if category == 'integration':
                self._run_integration_tests(verbose)
            elif category == 'performance':
                self._run_performance_tests(verbose)
            elif category == 'synthetic':
                self._run_synthetic_data_tests(verbose)
            elif category == 'memory':
                self._run_memory_tests(verbose)
            elif category == 'load':
                self._run_load_tests(verbose)
            else:
                print(f"Warning: Unknown test category '{category}'")
        
        self.end_time = time.time()
        
        # Generate final report
        final_report = self._generate_final_report()
        
        if self.report_file:
            self._save_detailed_report(final_report)
        
        return final_report
    
    def _run_integration_tests(self, verbose: bool = True):
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("INTEGRATION TESTS")
        print("=" * 60)
        
        if not COMPREHENSIVE_TESTS_AVAILABLE or not DARTS_AVAILABLE:
            print("⚠ Integration tests skipped - dependencies not available")
            self.test_results['integration'] = {'skipped': True, 'reason': 'Dependencies not available'}
            return
        
        try:
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveIntegration)
            
            # Run tests with custom result collector
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1,
                stream=sys.stdout,
                buffer=True
            ).run(suite)
            
            self.test_results['integration'] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
                'details': {
                    'failures': [str(f[1]) for f in result.failures],
                    'errors': [str(e[1]) for e in result.errors]
                }
            }
            
            print(f"\n📊 Integration Tests Summary:")
            print(f"   Tests run: {result.testsRun}")
            print(f"   Success rate: {self.test_results['integration']['success_rate']:.1%}")
            
        except Exception as e:
            print(f"✗ Integration tests failed: {e}")
            self.test_results['integration'] = {'error': str(e)}
    
    def _run_performance_tests(self, verbose: bool = True):
        """Run performance benchmark tests."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        if not COMPREHENSIVE_TESTS_AVAILABLE or not DARTS_AVAILABLE:
            print("⚠ Performance tests skipped - dependencies not available")
            self.test_results['performance'] = {'skipped': True, 'reason': 'Dependencies not available'}
            return
        
        try:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmarks)
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1,
                stream=sys.stdout,
                buffer=True
            ).run(suite)
            
            self.test_results['performance'] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
            }
            
            print(f"\n📊 Performance Tests Summary:")
            print(f"   Tests run: {result.testsRun}")
            print(f"   Success rate: {self.test_results['performance']['success_rate']:.1%}")
            
        except Exception as e:
            print(f"✗ Performance tests failed: {e}")
            self.test_results['performance'] = {'error': str(e)}
    
    def _run_synthetic_data_tests(self, verbose: bool = True):
        """Run synthetic data generator tests."""
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA TESTS")
        print("=" * 60)
        
        if not COMPREHENSIVE_TESTS_AVAILABLE or not DARTS_AVAILABLE:
            print("⚠ Synthetic data tests skipped - dependencies not available")
            self.test_results['synthetic'] = {'skipped': True, 'reason': 'Dependencies not available'}
            return
        
        try:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestSyntheticDataGenerators)
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1,
                stream=sys.stdout,
                buffer=True
            ).run(suite)
            
            self.test_results['synthetic'] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
            }
            
            print(f"\n📊 Synthetic Data Tests Summary:")
            print(f"   Tests run: {result.testsRun}")
            print(f"   Success rate: {self.test_results['synthetic']['success_rate']:.1%}")
            
        except Exception as e:
            print(f"✗ Synthetic data tests failed: {e}")
            self.test_results['synthetic'] = {'error': str(e)}
    
    def _run_memory_tests(self, verbose: bool = True):
        """Run memory profiling tests."""
        print("\n" + "=" * 60)
        print("MEMORY PROFILING TESTS")
        print("=" * 60)
        
        if not COMPREHENSIVE_TESTS_AVAILABLE or not DARTS_AVAILABLE or not PSUTIL_AVAILABLE:
            print("⚠ Memory tests skipped - dependencies not available")
            self.test_results['memory'] = {'skipped': True, 'reason': 'Dependencies not available'}
            return
        
        try:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryProfiling)
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1,
                stream=sys.stdout,
                buffer=True
            ).run(suite)
            
            self.test_results['memory'] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
            }
            
            print(f"\n📊 Memory Tests Summary:")
            print(f"   Tests run: {result.testsRun}")
            print(f"   Success rate: {self.test_results['memory']['success_rate']:.1%}")
            
        except Exception as e:
            print(f"✗ Memory tests failed: {e}")
            self.test_results['memory'] = {'error': str(e)}
    
    def _run_load_tests(self, verbose: bool = True):
        """Run load testing tests."""
        print("\n" + "=" * 60)
        print("LOAD TESTING")
        print("=" * 60)
        
        if not COMPREHENSIVE_TESTS_AVAILABLE or not DARTS_AVAILABLE:
            print("⚠ Load tests skipped - dependencies not available")
            self.test_results['load'] = {'skipped': True, 'reason': 'Dependencies not available'}
            return
        
        try:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestLoadTesting)
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1,
                stream=sys.stdout,
                buffer=True
            ).run(suite)
            
            self.test_results['load'] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
            }
            
            print(f"\n📊 Load Tests Summary:")
            print(f"   Tests run: {result.testsRun}")
            print(f"   Success rate: {self.test_results['load']['success_rate']:.1%}")
            
        except Exception as e:
            print(f"✗ Load tests failed: {e}")
            self.test_results['load'] = {'error': str(e)}
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_execution_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate overall statistics
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        categories_run = 0
        
        for category, results in self.test_results.items():
            if 'error' in results or results.get('skipped', False):
                continue
            
            categories_run += 1
            total_tests += results.get('tests_run', 0)
            total_failures += results.get('failures', 0)
            total_errors += results.get('errors', 0)
            total_skipped += results.get('skipped', 0)
        
        overall_success_rate = (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
        
        final_report = {
            'execution_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'total_execution_time': total_execution_time,
                'categories_run': categories_run
            },
            'overall_statistics': {
                'total_tests': total_tests,
                'total_failures': total_failures,
                'total_errors': total_errors,
                'total_skipped': total_skipped,
                'overall_success_rate': overall_success_rate
            },
            'category_results': self.test_results,
            'system_info': {
                'darts_available': DARTS_AVAILABLE,
                'psutil_available': PSUTIL_AVAILABLE,
                'comprehensive_tests_available': COMPREHENSIVE_TESTS_AVAILABLE
            }
        }
        
        return final_report
    
    def _save_detailed_report(self, report: Dict[str, Any]):
        """Save detailed report to file."""
        try:
            with open(self.report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📁 Detailed report saved to: {self.report_file}")
        except Exception as e:
            print(f"⚠ Could not save report to {self.report_file}: {e}")
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final summary of all tests."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUITE - FINAL SUMMARY")
        print("=" * 80)
        
        exec_info = report['execution_info']
        stats = report['overall_statistics']
        
        print(f"Execution time: {exec_info['total_execution_time']:.1f} seconds")
        print(f"Categories run: {exec_info['categories_run']}")
        print(f"Total tests: {stats['total_tests']}")
        print(f"Overall success rate: {stats['overall_success_rate']:.1%}")
        
        if stats['total_failures'] > 0:
            print(f"⚠ Failures: {stats['total_failures']}")
        
        if stats['total_errors'] > 0:
            print(f"✗ Errors: {stats['total_errors']}")
        
        if stats['total_skipped'] > 0:
            print(f"⏭ Skipped: {stats['total_skipped']}")
        
        print("\nCategory Breakdown:")
        for category, results in report['category_results'].items():
            if results.get('skipped', False):
                print(f"  {category:12s}: SKIPPED ({results.get('reason', 'Unknown reason')})")
            elif 'error' in results:
                print(f"  {category:12s}: ERROR")
            else:
                success_rate = results.get('success_rate', 0)
                status = "✓" if success_rate >= 0.8 else "⚠" if success_rate >= 0.5 else "✗"
                print(f"  {category:12s}: {status} {success_rate:.1%} ({results.get('tests_run', 0)} tests)")
        
        # Overall assessment
        print("\n" + "=" * 80)
        if stats['overall_success_rate'] >= 0.9:
            print("🎉 COMPREHENSIVE TEST SUITE: EXCELLENT - System is highly reliable")
        elif stats['overall_success_rate'] >= 0.8:
            print("✅ COMPREHENSIVE TEST SUITE: GOOD - System is reliable with minor issues")
        elif stats['overall_success_rate'] >= 0.6:
            print("⚠ COMPREHENSIVE TEST SUITE: FAIR - System has some reliability concerns")
        else:
            print("❌ COMPREHENSIVE TEST SUITE: POOR - System has significant reliability issues")
        
        print("=" * 80)


def main():
    """Main entry point for comprehensive test runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for DARTS Stock Forecasting System')
    
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=['integration', 'performance', 'synthetic', 'memory', 'load'],
        default=['integration', 'performance', 'synthetic', 'memory', 'load'],
        help='Test categories to run'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for detailed results (JSON format)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (integration and synthetic)'
    )
    
    args = parser.parse_args()
    
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Adjust categories for quick mode
    if args.quick:
        args.categories = ['integration', 'synthetic']
        print("Quick mode: Running integration and synthetic tests only")
    
    # Set default output file if not specified
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'comprehensive_test_results_{timestamp}.json'
    
    # Run comprehensive tests
    runner = ComprehensiveTestRunner()
    
    try:
        report = runner.run_test_suite(
            test_categories=args.categories,
            output_file=args.output,
            verbose=not args.quiet
        )
        
        # Print final summary
        runner.print_final_summary(report)
        
        # Exit with appropriate code
        overall_success_rate = report['overall_statistics']['overall_success_rate']
        if overall_success_rate >= 0.8:
            sys.exit(0)  # Success
        elif overall_success_rate >= 0.5:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Failure
            
    except KeyboardInterrupt:
        print("\n\n⚠ Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n✗ Test execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()