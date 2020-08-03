PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE Student (
id integer primary key autoincrement not null,
name text not null default 'blank',
mobile text null);
INSERT INTO Student VALUES(1,'홍길동','010-4949-1735');
INSERT INTO Student VALUES(2,'홍길순',NULL);
DELETE FROM sqlite_sequence;
INSERT INTO sqlite_sequence VALUES('Student',2);
COMMIT;
