/*
 * Kilo - A simple text editor with extended features
 * Based on the original Kilo by antirez
 * Features added: Search, Line Numbers, Undo, Auto-indent
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

/*** defines ***/

#define KILO_VERSION "0.0.1"
#define TAB_STOP 8
#define KILO_QUIT_TIMES 3

#define CTRL_KEY(k) ((k) & 0x1f)

enum editorKey {
    BACKSPACE = 127,
    ARROW_LEFT = 1000,
    ARROW_RIGHT,
    ARROW_UP,
    ARROW_DOWN,
    DEL_KEY,
    HOME_KEY,
    END_KEY,
    PAGE_UP,
    PAGE_DOWN
};

/*** data ***/

typedef struct erow {
    int idx;
    int size;
    int rsize;
    char *chars;
    char *render;
} erow;

struct editorConfig {
    int cx, cy;
    int rx;
    int rowoff;
    int coloff;
    int screenrows;
    int screencols;
    int numrows;
    erow *row;
    int dirty;
    char *filename;
    char statusmsg[80];
    time_t statusmsg_time;
    struct termios orig_termios;
};

struct editorConfig E;

/*** undo buffer ***/
typedef struct {
    int row_idx;
    char *text;
    int text_size;
    int type; /* 0 = insert, 1 = delete */
} undo_entry;

#define UNDO_BUFFER_SIZE 100
undo_entry undo_buffer[UNDO_BUFFER_SIZE];
int undo_count = 0;
int undo_index = -1; /* Current position in undo buffer */

/*** search state ***/
char search_query[80];
int search_last_match;
int search_direction = 1; /* 1 = forward, -1 = backward */
int search_saved_cx, search_saved_cy;

/*** prototypes ***/

void editorSetStatusMessage(const char *fmt, ...);
void editorRefreshScreen(void);
char *editorPrompt(char *prompt, void (*callback)(char *, int));

/*** terminal ***/

void die(const char *s) {
    write(STDOUT_FILENO, "\x1b[2J", 4);
    write(STDOUT_FILENO, "\x1b[H", 3);
    perror(s);
    exit(1);
}

void disableRawMode(void) {
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &E.orig_termios) == -1)
        die("tcsetattr");
}

void enableRawMode(void) {
    if (tcgetattr(STDIN_FILENO, &E.orig_termios) == -1) die("tcgetattr");
    atexit(disableRawMode);
    struct termios raw = E.orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == -1) die("tcsetattr");
}

int editorReadKey(void) {
    int nread;
    char c;
    while ((nread = read(STDIN_FILENO, &c, 1)) != 1) {
        if (nread == -1 && errno != EAGAIN) die("read");
    }
    if (c == '\x1b') {
        char seq[3];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return '\x1b';
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return '\x1b';
        if (seq[0] == '[') {
            if (seq[1] >= '0' && seq[1] <= '9') {
                if (read(STDIN_FILENO, &seq[2], 1) != 1) return '\x1b';
                if (seq[2] == '~') {
                    switch (seq[1]) {
                        case '1': return HOME_KEY;
                        case '3': return DEL_KEY;
                        case '4': return END_KEY;
                        case '5': return PAGE_UP;
                        case '6': return PAGE_DOWN;
                        case '7': return HOME_KEY;
                        case '8': return END_KEY;
                    }
                }
            } else {
                switch (seq[1]) {
                    case 'A': return ARROW_UP;
                    case 'B': return ARROW_DOWN;
                    case 'C': return ARROW_RIGHT;
                    case 'D': return ARROW_LEFT;
                    case 'H': return HOME_KEY;
                    case 'F': return END_KEY;
                }
            }
        } else if (seq[0] == 'O') {
            switch (seq[1]) {
                case 'H': return HOME_KEY;
                case 'F': return END_KEY;
            }
        }
        return '\x1b';
    } else {
        return c;
    }
}

int getCursorPosition(int *rows, int *cols) {
    char buf[32];
    unsigned int i = 0;
    if (write(STDOUT_FILENO, "\x1b[6n", 4) != 4) return -1;
    while (i < sizeof(buf) - 1) {
        if (read(STDIN_FILENO, &buf[i], 1) != 1) break;
        if (buf[i] == 'R') break;
        i++;
    }
    buf[i] = '\0';
    if (buf[0] != '\x1b' || buf[1] != '[') return -1;
    if (sscanf(buf + 2, "%d;%d", rows, cols) != 2) return -1;
    return 0;
}

int getWindowSize(int *rows, int *cols) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1 || ws.ws_col == 0) {
        if (write(STDOUT_FILENO, "\x1b[999C\x1b[999B", 12) != 12) return -1;
        return getCursorPosition(rows, cols);
    } else {
        *cols = ws.ws_col;
        *rows = ws.ws_row;
        return 0;
    }
}

/*** row operations ***/

int editorRowCxToRx(erow *row, int cx) {
    int rx = 0;
    int j;
    for (j = 0; j < cx; j++) {
        if (row->chars[j] == '\t')
            rx += (TAB_STOP - 1) - (rx % TAB_STOP);
        rx++;
    }
    return rx;
}

int editorRowRxToCx(erow *row, int rx) {
    int cx = 0;
    int curr_rx;
    for (curr_rx = 0; curr_rx < rx; curr_rx++) {
        if (row->chars[cx] == '\t') {
            curr_rx += (TAB_STOP - 1) - (curr_rx % TAB_STOP);
        }
        if (cx < row->size) cx++;
    }
    return cx;
}

void editorUpdateRow(erow *row) {
    int tabs = 0;
    int j;
    for (j = 0; j < row->size; j++)
        if (row->chars[j] == '\t') tabs++;
    free(row->render);
    row->render = malloc(row->size + tabs * (TAB_STOP - 1) + 1);
    int idx = 0;
    for (j = 0; j < row->size; j++) {
        if (row->chars[j] == '\t') {
            row->render[idx++] = ' ';
            while ((idx + 1) % TAB_STOP != 0) row->render[idx++] = ' ';
        } else {
            row->render[idx++] = row->chars[j];
        }
    }
    row->render[idx] = '\0';
    row->rsize = idx;
}

void editorInsertRow(int at, char *s, size_t len) {
    if (at < 0 || at > E.numrows) return;
    E.row = realloc(E.row, sizeof(erow) * (E.numrows + 1));
    memmove(&E.row[at + 1], &E.row[at], sizeof(erow) * (E.numrows - at));
    for (int j = at + 1; j <= E.numrows; j++) E.row[j].idx++;
    E.row[at].idx = at;
    E.row[at].size = len;
    E.row[at].chars = malloc(len + 1);
    memcpy(E.row[at].chars, s, len + 1);
    E.row[at].render = NULL;
    E.row[at].rsize = 0;
    editorUpdateRow(&E.row[at]);
    E.numrows++;
    E.dirty++;
}

void editorFreeRow(erow *row) {
    free(row->render);
    free(row->chars);
}

void editorDelRow(int at) {
    if (at < 0 || at >= E.numrows) return;
    editorFreeRow(&E.row[at]);
    memmove(&E.row[at], &E.row[at + 1], sizeof(erow) * (E.numrows - at - 1));
    for (int j = at; j < E.numrows - 1; j++) E.row[j].idx--;
    E.numrows--;
    E.dirty++;
}

void editorRowInsertChar(erow *row, int at, int c) {
    if (at < 0 || at > row->size) at = row->size;
    row->chars = realloc(row->chars, row->size + 2);
    memmove(&row->chars[at + 1], &row->chars[at], row->size - at + 1);
    row->size++;
    row->chars[at] = c;
    editorUpdateRow(row);
    E.dirty++;
}

void editorRowAppendString(erow *row, char *s, size_t len) {
    row->chars = realloc(row->chars, row->size + len + 1);
    memcpy(&row->chars[row->size], s, len);
    row->size += len;
    row->chars[row->size] = '\0';
    editorUpdateRow(row);
    E.dirty++;
}

void editorRowDeleteChar(erow *row, int at) {
    if (at < 0 || at >= row->size) return;
    memmove(&row->chars[at], &row->chars[at + 1], row->size - at);
    row->size--;
    editorUpdateRow(row);
    E.dirty++;
}

/*** undo operations ***/

void editorSaveUndoState(int row_idx, char *text, int text_size, int type) {
    if (undo_count >= UNDO_BUFFER_SIZE) {
        /* Shift buffer left to make room */
        for (int i = 0; i < UNDO_BUFFER_SIZE - 1; i++) {
            undo_buffer[i] = undo_buffer[i + 1];
        }
        undo_count = UNDO_BUFFER_SIZE - 1;
    }
    
    undo_buffer[undo_count].row_idx = row_idx;
    undo_buffer[undo_count].text = malloc(text_size + 1);
    memcpy(undo_buffer[undo_count].text, text, text_size);
    undo_buffer[undo_count].text[text_size] = '\0';
    undo_buffer[undo_count].text_size = text_size;
    undo_buffer[undo_count].type = type;
    undo_count++;
    undo_index = undo_count - 1;
}

void editorUndo(void) {
    if (undo_count == 0) {
        editorSetStatusMessage("Nothing to undo");
        return;
    }
    
    if (undo_index < 0) undo_index = 0;
    if (undo_index >= undo_count) undo_index = undo_count - 1;
    
    undo_entry *entry = &undo_buffer[undo_index];
    
    if (entry->type == 0) {
        /* Undo insertion: delete the text from the row */
        if (entry->row_idx >= 0 && entry->row_idx < E.numrows) {
            erow *row = &E.row[entry->row_idx];
            /* Delete the text that was inserted */
            for (int i = 0; i < entry->text_size; i++) {
                editorRowDeleteChar(row, row->size - 1);
            }
        }
    } else if (entry->type == 1) {
        /* Undo deletion: restore the text to the row */
        if (entry->row_idx >= 0 && entry->row_idx < E.numrows) {
            erow *row = &E.row[entry->row_idx];
            for (int i = 0; i < entry->text_size; i++) {
                editorRowInsertChar(row, row->size, entry->text[i]);
            }
        } else if (entry->row_idx == E.numrows) {
            /* Restoring a deleted row */
            editorInsertRow(E.numrows, entry->text, entry->text_size);
        }
    }
    
    undo_index--;
    E.dirty++;
    editorSetStatusMessage("Undid action");
}

/*** editor operations ***/

void editorInsertChar(int c) {
    if (E.cy == E.numrows) {
        editorInsertRow(E.numrows, "", 0);
    }
    
    /* Save state for undo before insertion */
    char text_buf[2] = {c, '\0'};
    editorSaveUndoState(E.cy, text_buf, 1, 0);
    
    editorRowInsertChar(&E.row[E.cy], E.cx, c);
    E.cx++;
}

void editorInsertNewline(void) {
    if (E.cx == 0) {
        /* Insert new row above current row */
        editorInsertRow(E.cy, "", 0);
    } else if (E.cx == E.row[E.cy].size) {
        /* Append new row after current row */
        editorInsertRow(E.cy + 1, "", 0);
    } else {
        /* Split row at cursor position */
        erow *row = &E.row[E.cy];
        
        /* Save for undo */
        char *remaining = malloc(row->size - E.cx + 1);
        memcpy(remaining, &row->chars[E.cx], row->size - E.cx);
        remaining[row->size - E.cx] = '\0';
        editorSaveUndoState(E.cy, remaining, row->size - E.cx, 1);
        free(remaining);
        
        editorInsertRow(E.cy + 1, &row->chars[E.cx], row->size - E.cx);
        row = &E.row[E.cy];
        row->size = E.cx;
        row->chars[row->size] = '\0';
        editorUpdateRow(row);
    }
    
    /* Auto-indentation: check if previous line has leading whitespace */
    if (E.cy > 0) {
        erow *prev_row = &E.row[E.cy - 1];
        int indent = 0;
        while (indent < prev_row->size && 
               (prev_row->chars[indent] == ' ' || prev_row->chars[indent] == '\t')) {
            indent++;
        }
        if (indent > 0) {
            /* Copy the indentation to the new line */
            char *indent_str = malloc(indent + 1);
            memcpy(indent_str, prev_row->chars, indent);
            indent_str[indent] = '\0';
            editorInsertRow(E.cy + 1, indent_str, indent);
            free(indent_str);
            E.cy++;
            E.cx = indent;
        } else {
            E.cy++;
        }
    } else {
        E.cy++;
    }
    E.cx = 0;
}

void editorDeleteChar(void) {
    if (E.cy == E.numrows) return;
    if (E.cx == 0 && E.cy == 0) return;
    
    erow *row = &E.row[E.cy];
    if (E.cx > 0) {
        /* Save for undo */
        char text_buf[2] = {row->chars[E.cx - 1], '\0'};
        editorSaveUndoState(E.cy, text_buf, 1, 1);
        
        editorRowDeleteChar(row, E.cx - 1);
        E.cx--;
    } else {
        /* At beginning of line: merge with previous line */
        E.cx = E.row[E.cy - 1].size;
        
        /* Save for undo */
        editorSaveUndoState(E.cy, row->chars, row->size, 1);
        
        editorRowAppendString(&E.row[E.cy - 1], row->chars, row->size);
        editorDelRow(E.cy);
        E.cy--;
    }
}

/*** search functionality ***/

void editorSearchCallback(char *query, int key) {
    if (key == '\r' || key == '\x1b') {
        search_direction = 1;
        return;
    } else if (key == ARROW_RIGHT || key == ARROW_DOWN) {
        search_direction = 1;
    } else if (key == ARROW_LEFT || key == ARROW_UP) {
        search_direction = -1;
    } else {
        search_direction = 1;
    }
    
    int current = E.cy;
    if (search_last_match == -1) {
        search_last_match = current;
    }
    
    for (int i = 0; i < E.numrows; i++) {
        current += search_direction;
        if (current < 0) current = E.numrows - 1;
        if (current >= E.numrows) current = 0;
        
        erow *row = &E.row[current];
        char *match = strstr(row->render, query);
        if (match) {
            search_last_match = current;
            E.cy = current;
            E.cx = editorRowRxToCx(row, match - row->render);
            E.rowoff = E.numrows;
            break;
        }
    }
}

void editorSearch(void) {
    search_saved_cx = E.cx;
    search_saved_cy = E.cy;
    search_last_match = -1;
    
    char *query = editorPrompt("Search: %s (ESC/Arrows/Enter)", editorSearchCallback);
    if (query) {
        free(query);
    } else {
        E.cx = search_saved_cx;
        E.cy = search_saved_cy;
    }
}

/*** file i/o ***/

char *editorRowsToString(int *buflen) {
    int totlen = 0;
    int j;
    for (j = 0; j < E.numrows; j++)
        totlen += E.row[j].size + 1;
    *buflen = totlen;
    char *buf = malloc(totlen);
    int len = 0;
    for (j = 0; j < E.numrows; j++) {
        memcpy(buf + len, E.row[j].chars, E.row[j].size);
        len += E.row[j].size;
        buf[len++] = '\n';
    }
    return buf;
}

void editorOpen(char *filename) {
    free(E.filename);
    E.filename = strdup(filename);
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        if (errno != ENOENT) die("fopen");
        return;
    }
    char *line = NULL;
    size_t linecap = 0;
    ssize_t linelen;
    while ((linelen = getline(&line, &linecap, fp)) != -1) {
        while (linelen > 0 && (line[linelen - 1] == '\n' ||
                               line[linelen - 1] == '\r'))
            linelen--;
        editorInsertRow(E.numrows, line, linelen);
    }
    free(line);
    fclose(fp);
    E.dirty = 0;
}

void editorSave(void) {
    if (E.filename == NULL) {
        E.filename = editorPrompt("Save as: %s", NULL);
        if (E.filename == NULL) {
            editorSetStatusMessage("Save aborted");
            return;
        }
    }
    int len;
    char *buf = editorRowsToString(&len);
    int fd = open(E.filename, O_RDWR | O_CREAT, 0644);
    if (fd != -1) {
        if (ftruncate(fd, len) != -1) {
            if (write(fd, buf, len) == len) {
                close(fd);
                free(buf);
                E.dirty = 0;
                editorSetStatusMessage("%d bytes written to disk", len);
                return;
            }
        }
        close(fd);
    }
    free(buf);
    editorSetStatusMessage("Can't save! I/O error: %s", strerror(errno));
}

/*** output ***/

void editorScroll(void) {
    E.rx = 0;
    if (E.cy < E.numrows) {
        E.rx = editorRowCxToRx(&E.row[E.cy], E.cx);
    }
    if (E.cy < E.rowoff) {
        E.rowoff = E.cy;
    }
    if (E.cy >= E.rowoff + E.screenrows) {
        E.rowoff = E.cy - E.screenrows + 1;
    }
    if (E.rx < E.coloff) {
        E.coloff = E.rx;
    }
    if (E.rx >= E.coloff + E.screencols) {
        E.coloff = E.rx - E.screencols + 1;
    }
}

void editorDrawStatusBar(struct abuf *ab) {
    abAppend(ab, "\x1b[7m", 4);
    char status[80], rstatus[80];
    int len = snprintf(status, sizeof(status), "%.20s - %d lines %s",
                       E.filename ? E.filename : "[No Name]", E.numrows,
                       E.dirty ? "(modified)" : "");
    int rlen = snprintf(rstatus, sizeof(rstatus), "%d/%d",
                        E.cy + 1, E.numrows);
    if (len > E.screencols) len = E.screencols;
    abAppend(ab, status, len);
    while (len < E.screencols) {
        if (E.screencols - len == rlen) {
            abAppend(ab, rstatus, rlen);
            break;
        } else {
            abAppend(ab, " ", 1);
            len++;
        }
    }
    abAppend(ab, "\x1b[m", 3);
    abAppend(ab, "\r\n", 2);
}

void editorDrawMessageBar(struct abuf *ab) {
    abAppend(ab, "\x1b[K", 3);
    int msglen = strlen(E.statusmsg);
    if (msglen > E.screencols) msglen = E.screencols;
    if (msglen && time(NULL) - E.statusmsg_time < 5)
        abAppend(ab, E.statusmsg, msglen);
}

void editorDrawRows(struct abuf *ab) {
    int y;
    for (y = 0; y < E.screenrows; y++) {
        int filerow = y + E.rowoff;
        if (filerow >= E.numrows) {
            if (E.numrows == 0 && y == E.screenrows / 3) {
                char welcome[80];
                int welcomelen = snprintf(welcome, sizeof(welcome),
                                          "Kilo editor -- version %s", KILO_VERSION);
                if (welcomelen > E.screencols) welcomelen = E.screencols;
                int padding = (E.screencols - welcomelen) / 2;
                if (padding) {
                    abAppend(ab, "~", 1);
                    padding--;
                }
                while (padding--) abAppend(ab, " ", 1);
                abAppend(ab, welcome, welcomelen);
            } else {
                abAppend(ab, "~", 1);
            }
        } else {
            /* Draw line numbers */
            int line_num = filerow + 1;
            char line_num_str[16];
            int line_num_len = snprintf(line_num_str, sizeof(line_num_str), "%d ", line_num);
            
            /* Syntax highlighting: check if this is a comment line */
            if (E.row[filerow].size >= 2 && 
                E.row[filerow].chars[0] == '/' && 
                E.row[filerow].chars[1] == '/') {
                abAppend(ab, "\x1b[30;42m", 8); /* Green background for comments */
            }
            
            abAppend(ab, line_num_str, line_num_len);
            
            /* Reset colors after line number */
            abAppend(ab, "\x1b[0m", 4);
            
            /* Draw the actual row content */
            int len = E.row[filerow].rsize - E.coloff;
            if (len < 0) len = 0;
            if (len > E.screencols - line_num_len) len = E.screencols - line_num_len;
            if (len > 0) {
                abAppend(ab, &E.row[filerow].render[E.coloff], len);
            }
        }
        abAppend(ab, "\x1b[K", 3);
        abAppend(ab, "\r\n", 2);
    }
}

struct abuf {
    char *b;
    int len;
};

#define ABUF_INIT {NULL, 0}

void abAppend(struct abuf *ab, const char *s, int len) {
    char *new = realloc(ab->b, ab->len + len);
    if (new == NULL) return;
    memcpy(&new[ab->len], s, len);
    ab->b = new;
    ab->len += len;
}

void abFree(struct abuf *ab) {
    free(ab->b);
}

void editorRefreshScreen(void) {
    editorScroll();
    struct abuf ab = ABUF_INIT;
    abAppend(&ab, "\x1b[?25l", 6); /* Hide cursor */
    abAppend(&ab, "\x1b[H", 3);
    editorDrawRows(&ab);
    editorDrawStatusBar(&ab);
    editorDrawMessageBar(&ab);
    abAppend(&ab, "\x1b[H", 3);
    
    /* Position cursor at correct location */
    int x_offset = E.coloff > 0 ? E.coloff : 0;
    int line_num_len = 0;
    if (E.cy >= 0 && E.cy < E.numrows) {
        int line_num = E.cy + 1;
        line_num_len = snprintf(NULL, 0, "%d ", line_num);
    }
    
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "\x1b[%d;%dH", 
                       (E.cy - E.rowoff) + 1 + 1, /* +1 for 1-based, +1 for status bar */
                       (E.rx - E.coloff) + 1 + line_num_len); /* +1 for 1-based, +line_num_len for line numbers */
    abAppend(&ab, buf, len);
    
    abAppend(&ab, "\x1b[?25h", 6); /* Show cursor */
    write(STDOUT_FILENO, ab.b, ab.len);
    abFree(&ab);
}

void editorSetStatusMessage(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(E.statusmsg, sizeof(E.statusmsg), fmt, ap);
    va_end(ap);
    E.statusmsg_time = time(NULL);
}

char *editorPrompt(char *prompt, void (*callback)(char *, int)) {
    size_t bufsize = 128;
    char *buf = malloc(bufsize);
    if (!buf) die("malloc");
    size_t buflen = 0;
    buf[0] = '\0';
    
    while (1) {
        editorSetStatusMessage(prompt, buf);
        editorRefreshScreen();
        
        int c = editorReadKey();
        if (c == DEL_KEY || c == CTRL_KEY('h') || c == BACKSPACE) {
            if (buflen != 0) buf[--buflen] = '\0';
        } else if (c == '\x1b') {
            editorSetStatusMessage("");
            if (callback) callback(buf, c);
            free(buf);
            return NULL;
        } else if (c == '\r') {
            if (buflen != 0) {
                editorSetStatusMessage("");
                if (callback) callback(buf, c);
                return buf;
            }
        } else if (!iscntrl(c) && c < 128) {
            if (buflen == bufsize - 1) {
                bufsize *= 2;
                buf = realloc(buf, bufsize);
                if (!buf) die("realloc");
            }
            buf[buflen++] = c;
            buf[buflen] = '\0';
        }
        
        if (callback) callback(buf, c);
    }
}

/*** input ***/

void editorProcessKeypress(void) {
    static int quit_times = KILO_QUIT_TIMES;
    
    int c = editorReadKey();
    
    switch (c) {
        case '\r': /* Enter */
            editorInsertNewline();
            break;
        
        case CTRL_KEY('q'): /* Quit */
            if (E.dirty && quit_times > 0) {
                editorSetStatusMessage("WARNING!!! File has unsaved changes. "
                    "Press Ctrl-Q %d more times to quit.", quit_times);
                quit_times--;
                return;
            }
            write(STDOUT_FILENO, "\x1b[2J", 4);
            write(STDOUT_FILENO, "\x1b[H", 3);
            exit(0);
            break;
        
        case CTRL_KEY('s'): /* Save */
            editorSave();
            break;
        
        case CTRL_KEY('f'): /* Search */
            editorSearch();
            break;
        
        case CTRL_KEY('z'): /* Undo */
            editorUndo();
            break;
        
        case HOME_KEY:
            E.cx = 0;
            break;
        
        case END_KEY:
            if (E.cy < E.numrows)
                E.cx = E.row[E.cy].size;
            break;
        
        case BACKSPACE:
        case CTRL_KEY('h'):
        case DEL_KEY:
            if (c == DEL_KEY && E.cx != E.row[E.cy].size)
                E.cx++;
            editorDeleteChar();
            break;
        
        case PAGE_UP:
        case PAGE_DOWN:
            if (c == PAGE_UP) {
                E.cy = E.rowoff;
            } else if (c == PAGE_DOWN) {
                E.cy = E.rowoff + E.screenrows - 1;
                if (E.cy > E.numrows) E.cy = E.numrows;
            }
            
            int times = E.screenrows;
            while (times--)
                editorMoveCursor(c == PAGE_UP ? ARROW_UP : ARROW_DOWN);
            break;
        
        case ARROW_UP:
        case ARROW_DOWN:
        case ARROW_LEFT:
        case ARROW_RIGHT:
            editorMoveCursor(c);
            break;
        
        case CTRL_KEY('l'):
        case '\x1b':
            break;
        
        default:
            editorInsertChar(c);
            break;
    }
    
    quit_times = KILO_QUIT_TIMES;
}

void editorMoveCursor(int key) {
    erow *row = (E.cy >= E.numrows) ? NULL : &E.row[E.cy];
    
    switch (key) {
        case ARROW_LEFT:
            if (E.cx != 0) {
                E.cx--;
            } else if (E.cy > 0) {
                E.cy--;
                E.cx = E.row[E.cy].size;
            }
            break;
        case ARROW_RIGHT:
            if (row && E.cx < row->size) {
                E.cx++;
            } else if (row && E.cx == row->size && E.cy + 1 < E.numrows) {
                E.cy++;
                E.cx = 0;
            }
            break;
        case ARROW_UP:
            if (E.cy != 0) {
                E.cy--;
            }
            break;
        case ARROW_DOWN:
            if (E.cy < E.numrows) {
                E.cy++;
            }
            break;
    }
    
    row = (E.cy >= E.numrows) ? NULL : &E.row[E.cy];
    int rowlen = row ? row->size : 0;
    if (E.cx > rowlen) {
        E.cx = rowlen;
    }
}

/*** init ***/

void initEditor(void) {
    E.cx = 0;
    E.cy = 0;
    E.rx = 0;
    E.rowoff = 0;
    E.coloff = 0;
    E.numrows = 0;
    E.row = NULL;
    E.dirty = 0;
    E.filename = NULL;
    E.statusmsg[0] = '\0';
    E.statusmsg_time = 0;
    E.screenrows = 0;
    E.screencols = 0;
    
    if (getWindowSize(&E.screenrows, &E.screencols) == -1) die("getWindowSize");
    E.screenrows -= 2;
}

int main(int argc, char *argv[]) {
    enableRawMode();
    initEditor();
    
    if (argc >= 2) {
        editorOpen(argv[1]);
    }
    
    editorSetStatusMessage("HELP: Ctrl-Q = quit | Ctrl-S = save | Ctrl-F = search | Ctrl-Z = undo");
    
    while (1) {
        editorRefreshScreen();
        editorProcessKeypress();
    }
    
    return 0;
}
