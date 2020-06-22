from Engine import Engine,confirm
from CaseLibraryController import FIRST_CASE_LIBRARY_PATH, UPDATED_CASE_LIBRARY_PATH

if __name__ == '__main__':
    db_path = UPDATED_CASE_LIBRARY_PATH if confirm("Use the last updated database?") else FIRST_CASE_LIBRARY_PATH
    if db_path == FIRST_CASE_LIBRARY_PATH:
        print("If you decide to update the results when the session ends, all previous updates will be override")
    Engine(case_library_path=db_path, verbose=True).run()