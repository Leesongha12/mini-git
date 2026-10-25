"""python -m unittest -v 로 실행하는 표준 라이브러리 테스트."""
import ast
import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from main import (MiniGit, Commit, InvertedIndex, execute, merge_sort,
                  topological_order, shortest_path, find_ancestors)


class MiniGitTests(unittest.TestCase):
    def setUp(self):
        self.repo = MiniGit()

    def run_command(self, line):
        output = io.StringIO()
        with redirect_stdout(output):
            result = execute(self.repo, line)
        return result, output.getvalue()

    def graph(self):
        for line in ['init "Alice Kim"', 'commit "Initial commit"',
                     'branch feature', 'switch feature', 'commit "Add login feature"',
                     'switch main', 'commit "Add payment feature"', 'merge feature']:
            self.run_command(line)
        return [f'{i:016x}' for i in range(1, 5)]

    def test_topology_and_merge(self):
        ids = self.graph()
        order = topological_order(self.repo.commits, self.repo.children)
        for h, c in self.repo.commits.items():
            for parent in c.parents:
                self.assertLess(order.index(parent), order.index(h))
        self.assertEqual(self.repo.commits[ids[3]].parents, (ids[2], ids[1]))
        self.assertEqual(self.repo.branches['feature'], ids[1])

    def test_shortest_path_tie(self):
        a, b, c, d = self.graph()
        self.assertEqual(shortest_path(self.repo.neighbors, a, d), [a, b, d])
        self.assertEqual(shortest_path(self.repo.neighbors, b, c), [b, a, c])
        self.assertEqual(shortest_path(self.repo.neighbors, a, a), [a])

    def test_disconnected(self):
        for line in ['init Alice', 'branch orphan', 'commit A', 'switch orphan', 'commit B']:
            self.run_command(line)
        _, output = self.run_command('path 0000000000000001 0000000000000002')
        self.assertEqual(output.strip(), 'No path')

    def test_ancestors(self):
        a, b, c, d = self.graph()
        result = find_ancestors(self.repo.commits, d)
        self.assertEqual(set(result), {a, b, c})
        self.assertEqual(len(result), 3)
        self.assertEqual(find_ancestors(self.repo.commits, a), [])

    def test_search(self):
        a, b, c, d = self.graph()
        self.assertEqual(self.repo.index.by_keyword('LOGIN feature'), [b])
        self.assertEqual(self.repo.index.by_keyword('log'), [])
        self.assertEqual(self.repo.index.by_keyword('missing'), [])
        self.assertEqual(self.repo.index.by_author('Alice Kim'), [a, b, c, d])
        self.assertEqual(self.repo.index.by_author('alice kim'), [])
        self.assertIn('Found 4', self.run_command('search --author="Alice Kim"')[1])

    def test_duplicate_tokens_and_multiple_authors(self):
        index = InvertedIndex()
        now = datetime.now(timezone.utc)
        index.add(Commit('a', 'HELLO hello', 'Alice', now, ()))
        index.add(Commit('b', 'Hello world', 'Bob', now, ()))
        self.assertEqual(index.by_keyword('hello'), ['a', 'b'])
        self.assertEqual(index.by_author('Bob'), ['b'])

    def test_sort_stability_and_keys(self):
        rows = [(2, 'a'), (1, 'b'), (2, 'c'), (1, 'd')]
        self.assertEqual(merge_sort(rows, key=lambda x: x[0]),
                         [(1, 'b'), (1, 'd'), (2, 'a'), (2, 'c')])
        self.assertEqual(merge_sort([], key=lambda x: x), [])
        self.assertEqual(merge_sort([3, 2, 1], key=lambda x: x), [1, 2, 3])
        self.assertEqual(merge_sort(['Zoe', 'Alice'], key=lambda x: x), ['Alice', 'Zoe'])

    def test_log_options(self):
        ids = self.graph()
        for option in ['date', 'author']:
            output = self.run_command('log --sort-by=' + option)[1]
            positions = [output.index(h) for h in ids]
            self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))

    def test_parser_and_errors(self):
        with self.assertRaisesRegex(ValueError, 'Repository not initialized'):
            self.run_command('log')
        self.run_command('iNiT "이 송하"')
        self.run_command('cOmMiT "첫 번째 커밋"')
        self.assertEqual(self.repo.user, '이 송하')
        for line in ['commit', 'commit ""', 'commit "broken', 'log extra',
                     'search --author=', 'search --bad=x', 'path a', 'exit x']:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, 'Invalid args'):
                self.run_command(line)
        for line, error in [('switch missing', 'Unknown branch'),
                            ('ancestors missing', 'Unknown commit'),
                            ('init Bob', 'Repository already initialized'),
                            ('branch main', 'Branch already exists'),
                            ('wat', 'Unknown command')]:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, error):
                self.run_command(line)
        self.assertFalse(self.run_command('QuIt')[0])

    def test_merge_errors(self):
        self.run_command('init Alice')
        self.run_command('branch empty')
        with self.assertRaisesRegex(ValueError, 'Cannot merge empty'):
            self.run_command('merge empty')
        self.run_command('commit A')
        self.run_command('branch same')
        with self.assertRaisesRegex(ValueError, 'Cannot merge identical'):
            self.run_command('merge same')

    def test_id_exhaustion_no_reuse(self):
        self.run_command('init Alice')
        self.repo.counter = (1 << 64) - 2
        self.run_command('commit last')
        self.assertEqual(self.repo.head, 'ffffffffffffffff')
        with self.assertRaisesRegex(ValueError, 'Commit ID space exhausted'):
            self.run_command('commit overflow')
        self.assertEqual(len(self.repo.commits), 1)

    def test_no_standard_sort_calls(self):
        tree = ast.parse(Path(__file__).with_name('main.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, 'sorted')
                if isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, 'sort')


if __name__ == '__main__':
    unittest.main()
