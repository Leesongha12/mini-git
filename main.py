"""메모리 기반 Mini Git. Python 3.10+, 외부 패키지 없음."""
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import shlex


def merge_sort(items, key):
    """직접 구현한 안정 병합 정렬: 시간 O(n log n), 추가 공간 O(n)."""
    if len(items) <= 1:
        return list(items)
    middle = len(items) // 2
    left = merge_sort(items[:middle], key)
    right = merge_sort(items[middle:], key)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


@dataclass(frozen=True)
class Commit:
    """생성 후 변경되지 않는 커밋 메타데이터."""
    hash: str
    message: str
    author: str
    timestamp: datetime
    parents: tuple[str, ...]


def topological_order(commits, children):
    """Kahn 알고리즘: 부모가 자식보다 먼저 출력되는 순서를 반환한다."""
    indegree = {h: len(c.parents) for h, c in commits.items()}
    ready = deque(h for h in commits if indegree[h] == 0)
    result = []
    while ready:
        current = ready.popleft()
        result.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(result) != len(commits):
        raise ValueError('Invalid graph: cycle detected')
    return result


def find_ancestors(commits, start):
    """반복형 DFS로 자기 자신을 제외한 모든 조상을 중복 없이 반환한다."""
    visited = {start}
    stack = [start]
    result = []
    while stack:
        current = stack.pop()
        for parent in commits[current].parents:
            if parent not in visited:
                visited.add(parent)
                result.append(parent)
                stack.append(parent)
    return result


def shortest_path(neighbors, start, target):
    """목표에서 BFS 후, 거리가 감소하는 사전순 최소 이웃을 선택한다."""
    distance = {target: 0}
    queue = deque([target])
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    if start not in distance:
        return None
    path = [start]
    current = start
    while current != target:
        best = None
        for neighbor in neighbors[current]:
            if distance.get(neighbor) == distance[current] - 1:
                if best is None or neighbor < best:
                    best = neighbor
        if best is None:
            raise ValueError('Invalid graph')
        path.append(best)
        current = best
    return path


class InvertedIndex:
    """키워드/작성자에서 커밋 ID 목록으로 연결하는 역색인."""
    def __init__(self):
        self.keywords = {}
        self.authors = {}

    def add(self, commit):
        """메시지의 중복 토큰은 한 번만 등록한다."""
        for token in set(commit.message.lower().split()):
            self.keywords.setdefault(token, []).append(commit.hash)
        self.authors.setdefault(commit.author, []).append(commit.hash)

    def by_keyword(self, query):
        """소문자 토큰 검색. 여러 토큰은 AND 조건으로 처리한다."""
        tokens = list(dict.fromkeys(query.lower().split()))
        if not tokens:
            return []
        postings = []
        for token in tokens:
            if token not in self.keywords:
                return []
            postings.append(self.keywords[token])
        ordered = merge_sort(postings, key=len)
        membership = [set(p) for p in ordered[1:]]
        return [h for h in ordered[0] if all(h in group for group in membership)]

    def by_author(self, author):
        """작성자는 대소문자를 포함해 정확히 일치해야 한다."""
        return list(self.authors.get(author, []))


class MiniGit:
    """브랜치, 커밋 DAG, 인접 목록과 역색인 관리."""
    def __init__(self):
        self.initialized = False
        self.user = ''
        self.current_branch = ''
        self.branches = {}
        self.commits = {}
        self.children = {}
        self.neighbors = {}
        self.index = InvertedIndex()
        self.counter = 0

    @property
    def head(self):
        """현재 브랜치가 가리키는 커밋 ID."""
        return self.branches[self.current_branch]

    def require_initialized(self):
        if not self.initialized:
            raise ValueError('Repository not initialized')

    def require_commit(self, h):
        if h not in self.commits:
            raise ValueError(f'Unknown commit: {h}')

    def require_branch(self, name):
        if name not in self.branches:
            raise ValueError(f'Unknown branch: {name}')

    def init(self, user):
        """재초기화는 기존 기록 보호를 위해 거부한다."""
        if self.initialized:
            raise ValueError('Repository already initialized')
        self.user = user
        self.current_branch = 'main'
        self.branches['main'] = None
        self.initialized = True
        print(f'Initialized repository.\nCurrent branch: main\nCurrent user: {user}')

    def branch(self, name):
        if name in self.branches:
            raise ValueError(f'Branch already exists: {name}')
        self.branches[name] = self.head
        print(f'Created branch: {name}')

    def switch(self, name):
        self.require_branch(name)
        self.current_branch = name
        print(f'Switched to branch: {name}')

    def create_commit(self, message, parents):
        """기존 커밋만 부모로 허용하고 불변 노드를 추가해 DAG를 유지한다."""
        parents = tuple(dict.fromkeys(parents))
        for parent in parents:
            self.require_commit(parent)
        if self.counter >= (1 << 64) - 1:
            raise ValueError('Commit ID space exhausted')
        self.counter += 1
        h = f'{self.counter:016x}'
        commit = Commit(h, message, self.user, datetime.now(timezone.utc), parents)
        self.commits[h] = commit
        self.children[h] = []
        self.neighbors[h] = list(parents)
        for parent in parents:
            self.children[parent].append(h)
            self.neighbors[parent].append(h)
        self.index.add(commit)
        self.branches[self.current_branch] = h
        print(f'[{self.current_branch} {h}] {message}')

    def commit(self, message):
        parents = [] if self.head is None else [self.head]
        self.create_commit(message, parents)

    def merge(self, name):
        """파일 비교 없이 두 HEAD를 부모로 하는 병합 커밋을 생성한다."""
        self.require_branch(name)
        target = self.branches[name]
        if self.head is None or target is None:
            raise ValueError('Cannot merge empty branch')
        if self.head == target:
            raise ValueError('Cannot merge identical heads')
        self.create_commit(f"Merge branch '{name}' into '{self.current_branch}'",
                           [self.head, target])

    def show_commits(self, hashes):
        """커밋 메타데이터와 현재 브랜치 포인터를 표시한다."""
        hashes = list(hashes)
        if not hashes:
            print('No commits')
            return
        labels = {}
        for name, h in self.branches.items():
            if h is not None:
                labels.setdefault(h, []).append(name)
        for h in hashes:
            c = self.commits[h]
            timestamp = c.timestamp.isoformat(timespec='microseconds')
            names = labels.get(h, [])
            label = f" [{', '.join(names)}]" if names else ''
            print(f'commit {h} ({c.author}, {timestamp}){label}\n  {c.message}')

    def log(self, sort_by=None):
        """기본은 전체 DAG 위상 순서, 옵션은 지정 속성 오름차순."""
        if sort_by is None:
            hashes = topological_order(self.commits, self.children)
        else:
            if sort_by not in ('date', 'author'):
                raise ValueError('Invalid args')
            key = (lambda c: c.timestamp) if sort_by == 'date' else (lambda c: c.author)
            hashes = [c.hash for c in merge_sort(list(self.commits.values()), key)]
        self.show_commits(hashes)

    def path(self, start, target):
        self.require_commit(start)
        self.require_commit(target)
        path = shortest_path(self.neighbors, start, target)
        print('No path' if path is None else 'Path: ' + '->'.join(path))

    def ancestors(self, h):
        self.require_commit(h)
        hashes = find_ancestors(self.commits, h)
        if hashes:
            self.show_commits(hashes)
        else:
            print('No ancestors')

    def search(self, query, by_author=False):
        hashes = self.index.by_author(query) if by_author else self.index.by_keyword(query)
        print(f'Found {len(hashes)} commit(s):')
        if hashes:
            self.show_commits(hashes)


HELP = '''
INIT "user name"                 BRANCH <branch_name>
SWITCH <branch_name>             COMMIT "commit message"
LOG                             LOG --sort-by=date|author
PATH <commit1> <commit2>          ANCESTORS <commit_hash>
SEARCH "keyword"                SEARCH --author="user name"
MERGE <branch_name>              HELP / EXIT / QUIT

Keywords: whole tokens, case-insensitive; multiple tokens use AND.
Commit IDs: enter the full 16-character ID.
'''


def execute(repo, line):
    """파싱/인자 검증/명령 실행. 종료 요청이면 False를 반환한다."""
    try:
        parts = shlex.split(line)
    except ValueError:
        raise ValueError('Invalid args') from None
    if not parts:
        return True
    command, args = parts[0].upper(), parts[1:]
    if command in ('EXIT', 'QUIT', 'HELP'):
        if args:
            raise ValueError('Invalid args')
        if command == 'HELP':
            print(HELP)
            return True
        return False
    known = {'INIT', 'BRANCH', 'SWITCH', 'COMMIT', 'LOG', 'PATH',
             'ANCESTORS', 'SEARCH', 'MERGE'}
    if command not in known:
        raise ValueError(f'Unknown command: {parts[0]}')
    if command == 'LOG':
        if len(args) > 1 or (args and args[0] not in ('--sort-by=date', '--sort-by=author')):
            raise ValueError('Invalid args')
    else:
        expected = 2 if command == 'PATH' else 1
        if len(args) != expected or any(not arg.strip() for arg in args):
            raise ValueError('Invalid args')
    if command == 'SEARCH' and args[0].startswith('--'):
        if not args[0].startswith('--author=') or not args[0][9:].strip():
            raise ValueError('Invalid args')
    if command == 'INIT':
        repo.init(args[0])
        return True
    repo.require_initialized()
    if command == 'BRANCH':
        repo.branch(args[0])
    elif command == 'SWITCH':
        repo.switch(args[0])
    elif command == 'COMMIT':
        repo.commit(args[0])
    elif command == 'LOG':
        repo.log(args[0].split('=', 1)[1] if args else None)
    elif command == 'PATH':
        repo.path(*args)
    elif command == 'ANCESTORS':
        repo.ancestors(args[0])
    elif command == 'SEARCH':
        if args[0].startswith('--author='):
            repo.search(args[0][9:], by_author=True)
        else:
            repo.search(args[0])
    elif command == 'MERGE':
        repo.merge(args[0])
    return True


def main():
    """사용자 오류를 출력하고 다음 명령을 계속 받는 REPL."""
    repo = MiniGit()
    print('Mini Git — type HELP for commands.')
    while True:
        try:
            line = input('mini-git> ')
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break
        try:
            if not execute(repo, line):
                print('Bye!')
                break
        except ValueError as error:
            print(error)


if __name__ == '__main__':
    main()
